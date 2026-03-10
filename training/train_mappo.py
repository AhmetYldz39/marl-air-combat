"""
train_mappo.py
==============
MAPPO (Multi-Agent PPO) eğitim döngüsü.

Mimari (Faz 1):
    Actor  : MLP 2×256 ReLU → Gaussian policy (continuous action)
    Critic : MLP 2×256 ReLU, centralized (global obs) → scalar value
    Buffer : RolloutBuffer — obs/action/logprob/reward/done/value
    Update : GAE → PPO clip loss + value loss + entropy bonus

Karşı takım: MultiHeuristicPolicy (sabit, eğitilmiyor)
Eğitilen takım: Blue (konfigürasyonla değiştirilebilir)

Kullanım:
    python train_mappo.py
    python train_mappo.py --config configs/config.yaml --seed 42

Çıktılar:
    checkpoints/mappo_ep{N}.pt   — periyodik checkpoint
    logs/train_log.csv           — episode bazlı metrik log
    logs/train_log.json          — aynı verinin JSON formatı

Bağımlılıklar:
    - torch >= 2.0
    - numpy
    - yaml
    - dogfight_env.py
    - heuristic_agent.py
    - normalization.py (obs_dim için)

Bu dosya değişirse etkilenen dosyalar:
    - evaluation/eval.py   (checkpoint yükleme formatı)
"""

import os
import sys
import csv
import json
import time
import argparse
import yaml
import numpy as np
from pathlib import Path
from copy import deepcopy

PROJECT_ROOT = Path(__file__).resolve().parent

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Normal
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    class _Stub:
        class Module: pass
    nn = _Stub  # type: ignore
sys.path.insert(0, str(PROJECT_ROOT))

from envs.dogfight_env import DogfightEnv, BLUE, RED
from agents.heuristic_agent import MultiHeuristicPolicy
from utils.normalization import Normalizer

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
TRAIN_TEAM = BLUE   # eğitilen takım
OPP_TEAM   = RED    # heuristic rakip




if not _TORCH_AVAILABLE:
    raise ImportError(
        "PyTorch bulunamadı. Kurulum: pip install torch\n"
        "RolloutBuffer için: from train_mappo import RolloutBuffer"
    )

from training.rollout_buffer import RolloutBuffer

# ===========================================================================
# Actor Network
# ===========================================================================

class MAPPOActor(nn.Module):
    """
    Gaussian policy (ctrl) + Bernoulli policy (fire).

    Giriş : normalize obs vektörü (obs_dim,)
    Çıkış :
        ctrl : da, de, dr, dt — Normal dağılım, tanh/sigmoid squash
        fire : 0/1             — Bernoulli head (collapse'e karşı)
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # Kontrol aksiyonları: da, de, dr, dt (4 adet Gaussian)
        self.mean_head = nn.Linear(hidden, action_dim - 1)
        self.log_std   = nn.Parameter(torch.zeros(action_dim - 1))
        # Fire: ayrı Bernoulli head — collapse'i önler
        self.fire_head = nn.Linear(hidden, 1)

        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)
        # Fire head: başlangıçta ~%30 ateş olasılığı (logit=-0.85)
        nn.init.constant_(self.fire_head.bias, -0.85)
        nn.init.orthogonal_(self.fire_head.weight, gain=0.01)

    def forward(self, obs: "torch.Tensor"):
        """
        Returns
        -------
        mean       : (batch, action_dim-1)  — da, de, dr, dt
        log_std    : (action_dim-1,)
        fire_logit : (batch, 1)             — Bernoulli logit
        """
        feat       = self.net(obs)
        mean       = self.mean_head(feat)
        fire_logit = self.fire_head(feat)
        return mean, self.log_std, fire_logit

    def get_dist(self, obs: "torch.Tensor"):
        """Returns (Normal dist for ctrl, Bernoulli dist for fire)"""
        mean, log_std, fire_logit = self.forward(obs)
        std       = log_std.exp().clamp(min=1e-4, max=2.0)
        ctrl_dist = Normal(mean, std)
        fire_dist = torch.distributions.Bernoulli(logits=fire_logit)
        return ctrl_dist, fire_dist

    @torch.no_grad()
    def act(self, obs: "torch.Tensor", deterministic: bool = False):
        """
        Tek adım aksiyon örnekleme.

        Returns
        -------
        action_raw : (action_dim,) — [ctrl(4), fire(1)]
        log_prob   : scalar
        """
        ctrl_dist, fire_dist = self.get_dist(obs)
        if deterministic:
            ctrl_raw = ctrl_dist.mean
            fire_raw = (fire_dist.probs > 0.5).float()
        else:
            ctrl_raw = ctrl_dist.sample()
            fire_raw = fire_dist.sample()
        ctrl_lp  = ctrl_dist.log_prob(ctrl_raw).sum(-1)
        fire_lp  = fire_dist.log_prob(fire_raw).sum(-1)
        raw      = torch.cat([ctrl_raw, fire_raw], dim=-1)
        log_prob = ctrl_lp + fire_lp
        return raw, log_prob

    @staticmethod
    def squash(raw: "torch.Tensor") -> "torch.Tensor":
        """
        Ham aksiyon → geçerli aralık:
            [:3] (da, de, dr) → tanh  → [-1, 1]
            [3]  (dt)         → sigmoid → [0, 1]
            [4]  (fire)       → binary (Bernoulli'den geliyor, dokunma)
        """
        out = raw.clone()
        out[..., :3] = torch.tanh(raw[..., :3])
        out[..., 3]  = torch.sigmoid(raw[..., 3])
        return out


# ===========================================================================
# Critic Network (Centralized)
# ===========================================================================

class MAPPOCritic(nn.Module):
    """
    Centralized value function.

    Giriş : global obs (tüm ajanların obs'u concat) (global_obs_dim,)
    Çıkış : scalar value
    """

    def __init__(self, global_obs_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for i, layer in enumerate(self.net):
            if isinstance(layer, nn.Linear):
                gain = 1.0 if i < len(self.net) - 1 else 0.01
                nn.init.orthogonal_(layer.weight, gain=gain)
                nn.init.zeros_(layer.bias)

    def forward(self, global_obs: "torch.Tensor") -> "torch.Tensor":
        return self.net(global_obs).squeeze(-1)

    @torch.no_grad()
    def act(self, global_obs: "torch.Tensor") -> float:
        return float(self.forward(global_obs).item())


class MAPPOTrainer:
    """
    MAPPO eğitim motoru.

    Parametreler config.yaml'dan okunur.
    """

    def __init__(self, config: dict, device: str = "auto"):
        self.config = config
        tr = config["training"]

        # Cihaz
        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        # Ortam
        self.env = DogfightEnv(config)
        self.env.seed(int(tr.get("seed", 42)))

        # Eğitilen ajan ID'leri (Blue takımı)
        self.train_ids = self.env.blue_ids
        self.n_agents  = len(self.train_ids)

        # Ajan indexi
        self.agent_idx = {aid: i for i, aid in enumerate(self.train_ids)}

        # Observation boyutları
        norm = Normalizer(config)
        n_tm = self.env.n_per_team - 1
        n_en = self.env.n_per_team
        self.obs_dim        = norm.obs_dim(n_tm, n_en)
        self.global_obs_dim = self.obs_dim * self.n_agents
        self.action_dim     = self.env.action_dim  # 5

        # Hyperparametreler
        self.gamma         = float(tr.get("gamma",          0.99))
        self.gae_lambda    = float(tr.get("gae_lambda",     0.95))
        self.clip_eps      = float(tr.get("clip_epsilon",   0.2))
        self.entropy_coeff = float(tr.get("entropy_coeff",  0.01))
        self.vf_coeff      = float(tr.get("value_loss_coeff", 0.5))
        self.max_grad_norm = float(tr.get("max_grad_norm",  0.5))
        self.n_steps       = int(tr.get("n_steps",          128))
        self.n_epochs      = int(tr.get("n_epochs",         4))
        self.minibatch     = int(tr.get("minibatch_size",   64))
        self.total_steps   = int(tr.get("total_timesteps",  10_000_000))
        self.lr_actor      = float(tr.get("lr_actor",       3e-4))
        self.lr_critic     = float(tr.get("lr_critic",      3e-4))

        hidden = int(tr.get("hidden_dim", 256))

        # Network'ler
        self.actor  = MAPPOActor(self.obs_dim, self.action_dim,
                                  hidden=hidden).to(self.device)
        self.critic = MAPPOCritic(self.global_obs_dim,
                                   hidden=hidden).to(self.device)

        # Optimizer (actor + critic ayrı)
        self.opt_actor  = optim.Adam(self.actor.parameters(),  lr=self.lr_actor)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        # Heuristic rakip
        team_map = {aid: ("blue" if "blue" in aid else "red")
                    for aid in self.env.agent_ids}
        self.opp_policy = MultiHeuristicPolicy(config, self.env.agent_ids,
                                                team_map)

        # Buffer
        self.buffer = RolloutBuffer(
            self.n_steps, self.n_agents,
            self.obs_dim, self.action_dim, self.global_obs_dim
        )

        # Logging
        log_cfg       = config.get("logging", {})
        self.log_dir  = Path(log_cfg.get("log_dir",        "logs/"))
        self.ckpt_dir = Path(log_cfg.get("checkpoint_dir", "checkpoints/"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval  = int(log_cfg.get("log_interval",        100))
        self.ckpt_interval = int(log_cfg.get("checkpoint_interval", 1000))

        # İstatistik
        self.global_step   = 0
        self.episode_count = 0
        self._ep_rewards   = []
        self._ep_wins      = []   # 1=win, 0=diğer
        self._ep_losses    = []   # 1=loss, 0=diğer
        self._ep_draws     = []   # 1=draw, 0=diğer
        self._ep_lengths   = []
        self._ep_reasons   = []   # "win" | "loss" | "draw" | "timeout"
        self._update_count = 0

        # Erken durdurma — Faz 1 geçiş kriterleri
        cur = config.get("curriculum", {}).get("phase1_to_2", {})
        self.early_stop_window   = int(cur.get("eval_window",       200))
        self.early_stop_interval = int(cur.get("eval_window",       200))
        self.early_stop_win_rate = float(cur.get("min_win_rate",    0.40))
        self.early_stop_kill     = float(cur.get("min_kill_per_ep", 0.80))
        self.early_stop_oob      = float(cur.get("max_oob_rate",    0.05))
        self._stopped_early      = False


        print(f"[MAPPO] Device      : {self.device}")
        print(f"[MAPPO] obs_dim     : {self.obs_dim}")
        print(f"[MAPPO] global_obs  : {self.global_obs_dim}")
        print(f"[MAPPO] action_dim  : {self.action_dim}")
        print(f"[MAPPO] n_agents    : {self.n_agents}")
        print(f"[MAPPO] Actor params: "
              f"{sum(p.numel() for p in self.actor.parameters()):,}")
        print(f"[MAPPO] Critic params: "
              f"{sum(p.numel() for p in self.critic.parameters()):,}")

    # -----------------------------------------------------------------------
    # Ana Eğitim Döngüsü
    # -----------------------------------------------------------------------

    def train(self):
        """Ana eğitim döngüsü."""
        obs_dict = self._reset_episode()
        ep_reward = {aid: 0.0 for aid in self.train_ids}
        ep_steps  = 0

        print(f"\n[MAPPO] Eğitim başlıyor — toplam {self.total_steps:,} adım\n")
        t_start = time.time()

        while self.global_step < self.total_steps:

            # ── Rollout toplama ────────────────────────────────────────
            self.buffer.reset()

            for _ in range(self.n_steps):

                # Eğitilen ajanlar için aksiyon
                actions_train, log_probs_train, values = \
                    self._collect_train_actions(obs_dict)

                # Heuristic rakip aksiyonları — kademeli güçlendirme
                # İlk 500k adım: %0 heuristic (rastgele)
                # 500k-1M adım: %0 → %50 heuristic
                # 1M+ adım: %100 heuristic
                state_dict   = self.env.get_all_states()
                opp_strength = float(np.clip(
                    (self.global_step - 500_000) / 500_000, 0.0, 1.0
                ))
                if opp_strength < 1.0:
                    heur_acts = self.opp_policy.act(state_dict)
                    rand_acts  = {aid: np.random.uniform(-1,1,5).astype(np.float32)
                                  for aid in self.opp_policy.agent_ids
                                  if aid not in self.train_ids}
                    actions_opp = {
                        aid: (opp_strength * heur_acts[aid]
                              + (1-opp_strength) * rand_acts[aid])
                        for aid in rand_acts
                    }
                else:
                    actions_opp = self.opp_policy.act(state_dict)

                # Tüm aksiyonları birleştir
                action_dict = {**actions_train, **actions_opp}

                # Global obs (critic girişi)
                global_obs = self._build_global_obs(obs_dict)

                # Adım
                next_obs, rew_dict, done_dict, info_dict = \
                    self.env.step(action_dict)

                # Buffer'a ekle
                self.buffer.add(
                    obs        = obs_dict,
                    actions    = actions_train,
                    log_probs  = log_probs_train,
                    rewards    = {aid: rew_dict[aid] for aid in self.train_ids},
                    dones      = done_dict,
                    values     = values,
                    global_obs = global_obs,
                    agent_ids  = self.train_ids,
                )

                # İstatistik
                for aid in self.train_ids:
                    ep_reward[aid] += rew_dict[aid]
                ep_steps += 1
                self.global_step += self.n_agents

                obs_dict = next_obs

                # Episode bitti mi?
                if done_dict["__all__"]:
                    winner = done_dict.get("winner", "draw")
                    is_win   = 1 if winner == TRAIN_TEAM else 0
                    is_loss  = 1 if winner == OPP_TEAM   else 0
                    is_draw  = 1 if winner == "draw"     else 0
                    # Timeout: max_steps doldu ve beraberlik
                    reason = (
                        "win"     if is_win  else
                        "loss"    if is_loss else
                        "timeout" if ep_steps >= self.env.max_steps - 1 else
                        "draw"
                    )

                    mean_rew = np.mean([ep_reward[aid]
                                        for aid in self.train_ids])
                    self._ep_rewards.append(mean_rew)
                    self._ep_wins.append(is_win)
                    self._ep_losses.append(is_loss)
                    self._ep_draws.append(is_draw)
                    self._ep_lengths.append(ep_steps)
                    self._ep_reasons.append(reason)
                    self.episode_count += 1

                    if self.episode_count % self.log_interval == 0:
                        self._log_progress(t_start)

                    if self.episode_count % self.ckpt_interval == 0:
                        self._save_checkpoint()

                    # Erken durdurma kontrolü
                    if (self.episode_count % self.early_stop_interval == 0
                            and self.episode_count >= self.early_stop_window):
                        if self._check_early_stop():
                            self._save_checkpoint(final=True)
                            return

                    obs_dict  = self._reset_episode()
                    ep_reward = {aid: 0.0 for aid in self.train_ids}
                    ep_steps  = 0

            # ── PPO Güncelleme ─────────────────────────────────────────
            self._update()

        # Eğitim sonu
        self._save_checkpoint(final=True)
        print(f"\n[MAPPO] Eğitim tamamlandı — {self.episode_count} episode, "
              f"{self.global_step:,} adım")

    # -----------------------------------------------------------------------
    # Erken Durdurma
    # -----------------------------------------------------------------------

    def _check_early_stop(self) -> bool:
        """
        Son early_stop_window episode bakarak Faz 1 kriterlerini kontrol et.
        True  -> kriterler saglandi, egitimi durdur.
        False -> devam et.
        """
        w = self.early_stop_window
        wins    = self._ep_wins[-w:]
        rewards = self._ep_rewards[-w:]

        win_rate    = float(np.mean(wins))
        mean_reward = float(np.mean(rewards))

        # Kill tahmini: w_kill=10 oldugu icin kill olan ep'de reward yuksek
        kill_per_ep = sum(1 for r in rewards if r > 5.0) / max(len(rewards), 1)

        # OOB/crash: loss olan ama timeout olmayan episode'lar
        # (gerçek kill/crash = loss + kısa episode)
        losses  = self._ep_losses[-w:]
        lengths = self._ep_lengths[-w:]
        oob_rate = sum(
            1 for loss, ln in zip(losses, lengths)
            if loss and ln < self.env.max_steps * 0.5
        ) / max(len(losses), 1)

        passed = (
            win_rate    >= self.early_stop_win_rate and
            kill_per_ep >= self.early_stop_kill     and
            oob_rate    <= self.early_stop_oob
        )

        print(
            f"\n[EarlyStop] Ep {self.episode_count} | "
            f"win={win_rate:.2f}(>={self.early_stop_win_rate}) | "
            f"kill/ep={kill_per_ep:.2f}(>={self.early_stop_kill}) | "
            f"oob={oob_rate:.2f}(<={self.early_stop_oob})"
        )
        if passed:
            self._stopped_early = True
            print("[EarlyStop] Faz 1 kriterleri saglandi — egitim durduruluyor!")
        else:
            print("[EarlyStop] Henuz saglanmadi — devam ediliyor.")
        return passed

    # -----------------------------------------------------------------------
    # Aksiyon Toplama
    # -----------------------------------------------------------------------

    def _collect_train_actions(self, obs_dict: dict) -> tuple:
        """
        Eğitilen ajanlar için aksiyon, log_prob ve value hesapla.

        Returns
        -------
        actions    : dict[str, np.ndarray]
        log_probs  : dict[str, float]
        values     : np.ndarray (n_agents,)
        """
        actions   = {}
        log_probs = {}
        values_np = np.zeros(self.n_agents, dtype=np.float32)

        # Critic için global obs
        # NaN/Inf guard — fizik ıraksırsa actor patlamasın
        for aid in self.train_ids:
            if not np.all(np.isfinite(obs_dict[aid])):
                obs_dict[aid] = np.zeros_like(obs_dict[aid])

        global_obs_t = torch.FloatTensor(
            self._build_global_obs(obs_dict)
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            value = self.critic(global_obs_t).squeeze().item()
        # Tüm ajanlar aynı global value paylaşır (centralized critic)
        values_np[:] = value

        # Blue heuristic (CRITICAL override için)
        if not hasattr(self, '_blue_heuristic'):
            from agents.heuristic_agent import HeuristicAgent
            self._blue_heuristic = {
                aid: HeuristicAgent(self.config, aid)
                for aid in self.train_ids
            }

        state_dict = self.env.get_all_states()

        for aid in self.train_ids:
            obs_t = torch.FloatTensor(obs_dict[aid]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                raw, lp = self.actor.act(obs_t)
            squashed = MAPPOActor.squash(raw.squeeze(0)).cpu().numpy()

            # CRITICAL override: zemin/stall/sınır tehlikesinde heuristic devreye girer
            s = state_dict[aid]
            critical = self._blue_heuristic[aid]._critical_recovery(s)
            if critical is not None:
                squashed = MAPPOActor.squash(
                    torch.FloatTensor(critical)
                ).numpy()

            actions[aid]   = squashed
            log_probs[aid] = float(lp.item())

        return actions, log_probs, values_np

    # -----------------------------------------------------------------------
    # PPO Güncelleme
    # -----------------------------------------------------------------------

    def _update(self):
        """GAE + PPO clip loss + value loss + entropy."""
        # Bootstrap value
        obs_dict    = self.env._build_obs_dict()  # mevcut obs
        global_obs_t = torch.FloatTensor(
            self._build_global_obs(obs_dict)
        ).unsqueeze(0).to(self.device)
        with torch.no_grad():
            last_val = self.critic(global_obs_t).squeeze().item()
        last_values = np.full(self.n_agents, last_val, dtype=np.float32)

        # GAE
        advantages, returns = self.buffer.compute_gae(
            last_values, self.gamma, self.gae_lambda
        )

        # Normalize advantage (tüm ajanlar birlikte)
        adv_flat = advantages.reshape(-1)
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
        advantages = adv_flat.reshape(advantages.shape)

        # Tensor dönüşüm
        T, A = self.n_steps, self.n_agents
        obs_t     = torch.FloatTensor(
            self.buffer.obs.reshape(T * A, self.obs_dim)).to(self.device)
        act_t     = torch.FloatTensor(
            self.buffer.actions.reshape(T * A, self.action_dim)).to(self.device)
        old_lp_t  = torch.FloatTensor(
            self.buffer.log_probs.reshape(T * A)).to(self.device)
        adv_t     = torch.FloatTensor(
            advantages.reshape(T * A)).to(self.device)
        ret_t     = torch.FloatTensor(
            returns.reshape(T * A)).to(self.device)
        # Global obs: her adım için, ajan sayısı kadar tekrar et
        gobs_t    = torch.FloatTensor(
            np.repeat(self.buffer.global_obs, A, axis=0)
        ).to(self.device)

        dataset_size = T * A
        indices      = np.arange(dataset_size)

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.minibatch):
                mb_idx = indices[start: start + self.minibatch]

                mb_obs    = obs_t[mb_idx]
                mb_act    = act_t[mb_idx]
                mb_old_lp = old_lp_t[mb_idx]
                mb_adv    = adv_t[mb_idx]
                mb_ret    = ret_t[mb_idx]
                mb_gobs   = gobs_t[mb_idx]

                # Yeni log_prob ve entropy — ctrl + fire ayrı
                ctrl_dist, fire_dist = self.actor.get_dist(mb_obs)
                mb_ctrl = mb_act[..., :4]   # da, de, dr, dt
                mb_fire = mb_act[..., 4:5]  # fire (binary)
                new_lp  = (ctrl_dist.log_prob(mb_ctrl).sum(-1)
                           + fire_dist.log_prob(mb_fire).sum(-1))
                entropy = (ctrl_dist.entropy().sum(-1).mean()
                           + fire_dist.entropy().sum(-1).mean())

                # PPO ratio
                ratio   = torch.exp(new_lp - mb_old_lp)
                clip_r  = torch.clamp(ratio, 1 - self.clip_eps,
                                              1 + self.clip_eps)
                actor_loss = -torch.min(ratio * mb_adv,
                                         clip_r * mb_adv).mean()

                # Value loss
                value_pred = self.critic(mb_gobs).squeeze(-1)
                value_loss = nn.functional.mse_loss(value_pred, mb_ret)

                # Toplam kayıp
                loss = (actor_loss
                        + self.vf_coeff * value_loss
                        - self.entropy_coeff * entropy)

                self.opt_actor.zero_grad()
                self.opt_critic.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(),
                                          self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(),
                                          self.max_grad_norm)
                self.opt_actor.step()
                self.opt_critic.step()

        self._update_count += 1

    # -----------------------------------------------------------------------
    # Yardımcı Metodlar
    # -----------------------------------------------------------------------

    def _reset_episode(self) -> dict:
        self.opp_policy.reset()
        return self.env.reset()

    def _build_global_obs(self, obs_dict: dict) -> np.ndarray:
        """Eğitilen ajanların obs'unu birleştir → (global_obs_dim,)"""
        return np.concatenate(
            [obs_dict[aid] for aid in self.train_ids], axis=0
        )

    def _log_progress(self, t_start: float):
        """Konsol + CSV log."""
        window    = min(self.log_interval, len(self._ep_rewards))
        mean_rew  = float(np.mean(self._ep_rewards[-window:]))
        win_rate  = float(np.mean(self._ep_wins[-window:]))
        loss_rate = float(np.mean(self._ep_losses[-window:]))
        draw_rate = float(np.mean(self._ep_draws[-window:]))
        mean_len  = float(np.mean(self._ep_lengths[-window:]))
        elapsed   = time.time() - t_start
        steps_sec = self.global_step / max(elapsed, 1)

        # Son window'daki bitiş sebepleri
        reasons   = self._ep_reasons[-window:]
        r_counts  = {r: reasons.count(r) for r in ["win","loss","draw","timeout"]}

        print(
            f"[Ep {self.episode_count:>6}] "
            f"step={self.global_step:>9,} | "
            f"rew={mean_rew:>7.2f} | "
            f"W={win_rate:.2f} L={loss_rate:.2f} D={draw_rate:.2f} | "
            f"len={mean_len:>5.0f} | "
            f"{steps_sec:>5.0f}sps"
        )
        print(
            f"{'':>10}bitiş: "
            f"win={r_counts['win']:>3} "
            f"loss={r_counts['loss']:>3} "
            f"draw={r_counts['draw']:>3} "
            f"timeout={r_counts['timeout']:>3}"
        )

        # CSV
        csv_path = self.log_dir / "train_log.csv"
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["episode", "global_step", "mean_reward",
                             "win_rate", "loss_rate", "draw_rate",
                             "mean_ep_len", "updates",
                             "n_win", "n_loss", "n_draw", "n_timeout"])
            w.writerow([
                self.episode_count, self.global_step,
                round(mean_rew, 4), round(win_rate, 4),
                round(loss_rate, 4), round(draw_rate, 4),
                round(mean_len, 1), self._update_count,
                r_counts["win"], r_counts["loss"],
                r_counts["draw"], r_counts["timeout"],
            ])

    def _save_checkpoint(self, final: bool = False):
        """Actor + Critic ağırlıklarını kaydet."""
        tag  = "final" if final else f"ep{self.episode_count}"
        path = self.ckpt_dir / f"mappo_{tag}.pt"
        torch.save({
            "episode":      self.episode_count,
            "global_step":  self.global_step,
            "actor":        self.actor.state_dict(),
            "critic":       self.critic.state_dict(),
            "opt_actor":    self.opt_actor.state_dict(),
            "opt_critic":   self.opt_critic.state_dict(),
            "config":       self.config,
        }, path)
        print(f"[MAPPO] Checkpoint kaydedildi: {path}")

    def load_checkpoint(self, path: str):
        """Checkpoint yükle (eval veya devam eğitimi için)."""
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.opt_actor.load_state_dict(ckpt["opt_actor"])
        self.opt_critic.load_state_dict(ckpt["opt_critic"])
        self.episode_count = ckpt.get("episode",     0)
        self.global_step   = ckpt.get("global_step", 0)
        print(f"[MAPPO] Checkpoint yüklendi: {path} "
              f"(ep={self.episode_count}, step={self.global_step:,})")


# ===========================================================================
# Entry Point
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--seed",   type=int, default=None)
    p.add_argument("--device", default="auto",
                   help="auto | cpu | cuda | cuda:0")
    p.add_argument("--resume", default=None,
                   help="Checkpoint yolu (devam eğitimi)")
    return p.parse_args()


def main():
    args = parse_args()

    config_path = PROJECT_ROOT / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if args.seed is not None:
        config["training"]["seed"] = args.seed

    trainer = MAPPOTrainer(config, device=args.device)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()