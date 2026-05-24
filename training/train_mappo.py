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
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1, closefd=False)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1, closefd=False)
import json
import time
import argparse
import yaml
import numpy as np
from pathlib import Path
from copy import deepcopy

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
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
from agents.opponent_pool import OpponentPool
from envs.aircraft_model import STATE_X, STATE_Y, STATE_H, STATE_ALIVE
from envs.geometry_utils import distance_3d, bearing_angle, wrap_to_pi
try:
    from models.opponent_model import OpponentModel
except ImportError:
    OpponentModel = None

from collections import deque
from models.om_net import (
    CentralizedOpponentModel, CentralizedRoleAssigner,
    EnemyHistoryBuffer, OMReplayBuffer, build_team_state, get_om_label,
    INTENT_DIM, ROLE_DIM, INTENT_DEFENSIVE, ROLE_PAIRS,
)

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
TRAIN_TEAM = BLUE   # eğitilen takım
OPP_TEAM   = RED    # heuristic rakip


# ===========================================================================
# Curriculum Manager
# ===========================================================================

class CurriculumManager:
    """
    4-fazlı curriculum yöneticisi.

    Dahili faz numaraları → gösterim adı → env modu:
      1 = Faz-1   : 1v1 WEZ-yakın (500-1500m)
      2 = Faz-1.5 : 1v1 kademeli mesafe (2000→4000m)
      3 = Faz-2   : 1v1 normal spawn
      4 = Faz-3   : 2v2 normal spawn

    Faz 1 kriterleri (son eval_window ep):
        kill/ep >= phase1_kill_thresh  AND  win_rate >= phase1_win_thresh
        + en az phase1_min_ep episode fazda

    Faz 1.5 davranışı:
        Her phase15_step_episodes ep'de spawn mesafesini +phase15_dist_step arttır.
        Son phase15_step_episodes ep'de kill/ep < phase15_pullback_thresh ise geri çek.
        Geçiş: kill/ep >= phase15_kill_thresh (son eval_window ep)

    Faz 2 geçiş kriteri:
        kill/ep >= phase2_kill_thresh (son eval_window ep)
    """

    PHASE_NAMES = {
        1: "Faz-1   (1v1 WEZ-close)",
        2: "Faz-1.5 (1v1 dinamik-dist)",
        3: "Faz-2   (1v1 normal)",
        4: "Faz-3   (2v2 normal)",
    }

    def __init__(self, config: dict):
        cur = config.get("curriculum_v2", {})
        self.phase = 1

        self.eval_window            = int(cur.get("eval_window",             100))

        # Faz 1
        self.phase1_kill_thresh     = float(cur.get("phase1_kill_threshold", 0.30))
        self.phase1_win_thresh      = float(cur.get("phase1_win_threshold",  0.55))
        self.phase1_min_ep          = int(cur.get("phase1_min_episodes",     100))

        # Faz 1.5
        self.phase15_dist_start     = float(cur.get("phase15_dist_start",   4000.0))
        self.phase15_dist_max       = float(cur.get("phase15_dist_max",    16000.0))
        self.phase15_dist_step      = float(cur.get("phase15_dist_step",    1000.0))
        self.phase15_step_eps       = int(cur.get("phase15_step_episodes",   100))
        self.phase15_pullback_thresh= float(cur.get("phase15_pullback_thresh", 0.10))
        self.phase15_kill_thresh    = float(cur.get("phase15_kill_threshold", 0.20))
        self.phase15_min_ep         = int(cur.get("phase15_min_episodes",   1200))

        # Faz 2
        self.phase2_kill_thresh     = float(cur.get("phase2_kill_threshold", 0.20))
        self.phase2_win_thresh      = float(cur.get("phase2_win_threshold",  0.40))
        self.phase2_sustain_window  = int(cur.get("phase2_sustain_window",   200))
        self.phase2_min_ep          = int(cur.get("phase2_min_episodes",     500))

        # Geçmiş
        self._kill_history: list    = []
        self._win_history:  list    = []
        self._ep_in_phase:  int     = 0

        # Faz 1.5 mevcut spawn mesafesi
        self.current_spawn_dist     = self.phase15_dist_start

    def record_episode(self, kills: float, is_win: int):
        """Episode sonu kill ve kazanma bilgisini kaydet."""
        self._kill_history.append(float(kills))
        self._win_history.append(float(is_win))
        self._ep_in_phase += 1

        # Faz 1.5: her phase15_step_eps'de mesafeyi güncelle
        if self.phase == 2 and self._ep_in_phase % self.phase15_step_eps == 0:
            window_kills = self._kill_history[-self.phase15_step_eps:]
            recent_kill  = float(np.mean(window_kills)) if window_kills else 0.0
            if recent_kill < self.phase15_pullback_thresh:
                self.current_spawn_dist = max(
                    self.phase15_dist_start,
                    self.current_spawn_dist - self.phase15_dist_step
                )
                print(f"[Curriculum] Faz-1.5: kill/ep={recent_kill:.3f} < "
                      f"{self.phase15_pullback_thresh} → mesafe geri cekiliyor: "
                      f"{self.current_spawn_dist:.0f}m")
            else:
                self.current_spawn_dist = min(
                    self.phase15_dist_max,
                    self.current_spawn_dist + self.phase15_dist_step
                )
                print(f"[Curriculum] Faz-1.5: kill/ep={recent_kill:.3f} → "
                      f"mesafe artiriliyor: {self.current_spawn_dist:.0f}m")

    def check_transition(self) -> bool:
        """
        Kriter sağlandıysa True döndürür.
        Faz 4'ten sonra asla True dönmez.
        """
        if self.phase >= 4:
            return False
        if len(self._kill_history) < self.eval_window:
            return False

        recent_kill = float(np.mean(self._kill_history[-self.eval_window:]))
        recent_win  = float(np.mean(self._win_history[-self.eval_window:]))

        if self.phase == 1:
            return (self._ep_in_phase >= self.phase1_min_ep
                    and recent_kill >= self.phase1_kill_thresh
                    and recent_win  >= self.phase1_win_thresh)
        if self.phase == 2:
            return (self._ep_in_phase >= self.phase15_min_ep
                    and recent_kill >= self.phase15_kill_thresh)
        if self.phase == 3:
            if self._ep_in_phase < self.phase2_min_ep:
                return False
            return recent_kill >= self.phase2_kill_thresh
        return False

    def advance(self):
        """Bir sonraki faza geç — geçmişi sıfırla."""
        self.phase += 1
        self._ep_in_phase = 0
        self._kill_history.clear()
        self._win_history.clear()
        # Faz 1.5 başlangıç mesafesini sıfırla
        if self.phase == 2:
            self.current_spawn_dist = self.phase15_dist_start

    def status_str(self) -> str:
        """Kısa durum satırı."""
        n = len(self._kill_history)
        w = min(n, self.eval_window)
        recent_kill = float(np.mean(self._kill_history[-w:])) if n > 0 else 0.0
        recent_win  = float(np.mean(self._win_history[-w:]))  if n > 0 else 0.0

        if self.phase == 1:
            thresh_str = (f"kill>={self.phase1_kill_thresh:.2f} "
                          f"win>={self.phase1_win_thresh:.2f}")
        elif self.phase == 2:
            thresh_str = (f"kill>={self.phase15_kill_thresh:.2f} "
                          f"dist={self.current_spawn_dist:.0f}m")
        elif self.phase == 3:
            thresh_str = (f"kill>={self.phase2_kill_thresh:.2f} "
                          f"min={self.phase2_min_ep}")
        else:
            thresh_str = "son faz"

        return (f"{self.PHASE_NAMES.get(self.phase, f'Faz-{self.phase}')} | "
                f"ep_in_phase={self._ep_in_phase} | "
                f"kill/ep={recent_kill:.3f} win={recent_win:.3f} "
                f"[{thresh_str}]")




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
            [:3] (da, de, dr) → clamp(-1, 1)
            [3]  (dt)         → clamp( 0, 1)
            [4]  (fire)       → binary (Bernoulli'den geliyor, dokunma)
        """
        out = raw.clone()
        out[..., :3] = torch.clamp(raw[..., :3], -1.0, 1.0)
        out[...,  3] = torch.clamp(raw[...,  3],  0.0, 1.0)
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


# ===========================================================================
# Faz-2 Transfer Learning Ağları
# ===========================================================================

class GATMAPPOActor(nn.Module):
    """
    Transfer learning Actor — 4-yönlü split giriş.

    Obs yapısı (78D):
      [0:50]   fc1_old    — base obs (checkpoint'ten)
      [50:68]  fc1_new    — eski ext (2D dummy + 16D GAT, checkpoint'ten)
      [68:74]  fc1_intent — yeni OM intent 6D (sıfır init)
      [74:78]  fc1_role   — yeni rol ataması 4D (sıfır init)
    """

    def __init__(self, old_obs_dim: int, ext_dim: int,
                 intent_dim: int, role_dim: int,
                 action_dim: int, hidden: int = 256):
        super().__init__()
        self.old_obs_dim  = old_obs_dim   # 50
        self.ext_dim      = ext_dim       # 18
        self.intent_dim   = intent_dim    # 6
        self.role_dim     = role_dim      # 4
        self._s0 = old_obs_dim
        self._s1 = old_obs_dim + ext_dim
        self._s2 = old_obs_dim + ext_dim + intent_dim

        self.fc1_old    = nn.Linear(old_obs_dim,  hidden, bias=True)
        self.fc1_new    = nn.Linear(ext_dim,      hidden, bias=False)
        self.fc1_intent = nn.Linear(intent_dim,   hidden, bias=False)
        self.fc1_role   = nn.Linear(role_dim,     hidden, bias=False)
        self.fc2        = nn.Linear(hidden, hidden)
        self.mean_head  = nn.Linear(hidden, action_dim - 1)
        self.log_std    = nn.Parameter(torch.zeros(action_dim - 1))
        self.fire_head  = nn.Linear(hidden, 1)

        self._init_new_weights()

    def _init_new_weights(self):
        nn.init.zeros_(self.fc1_new.weight)
        nn.init.zeros_(self.fc1_intent.weight)
        nn.init.zeros_(self.fc1_role.weight)
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.zeros_(self.fc2.bias)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.constant_(self.fire_head.bias, -0.85)
        nn.init.orthogonal_(self.fire_head.weight, gain=0.01)

    def freeze_old(self):
        self.fc1_old.requires_grad_(False)
        self.fc1_new.requires_grad_(False)

    def unfreeze_old(self):
        self.fc1_old.requires_grad_(True)
        self.fc1_new.requires_grad_(True)

    def forward(self, obs: "torch.Tensor"):
        feat = F.relu(
            self.fc1_old(obs[..., :self._s0]) +
            self.fc1_new(obs[..., self._s0:self._s1]) +
            self.fc1_intent(obs[..., self._s1:self._s2]) +
            self.fc1_role(obs[..., self._s2:])
        )
        feat = F.relu(self.fc2(feat))
        return self.mean_head(feat), self.log_std, self.fire_head(feat)

    def get_dist(self, obs: "torch.Tensor"):
        mean, log_std, fire_logit = self.forward(obs)
        std       = log_std.exp().clamp(min=1e-4, max=2.0)
        ctrl_dist = Normal(mean, std)
        fire_dist = torch.distributions.Bernoulli(logits=fire_logit)
        return ctrl_dist, fire_dist

    @torch.no_grad()
    def act(self, obs: "torch.Tensor", deterministic: bool = False):
        ctrl_dist, fire_dist = self.get_dist(obs)
        if deterministic:
            ctrl_raw = ctrl_dist.mean
            fire_raw = (fire_dist.probs > 0.5).float()
        else:
            ctrl_raw = ctrl_dist.sample()
            fire_raw = fire_dist.sample()
        ctrl_lp = ctrl_dist.log_prob(ctrl_raw).sum(-1)
        fire_lp = fire_dist.log_prob(fire_raw).sum(-1)
        raw     = torch.cat([ctrl_raw, fire_raw], dim=-1)
        return raw, ctrl_lp + fire_lp

    @staticmethod
    def squash(raw: "torch.Tensor") -> "torch.Tensor":
        out = raw.clone()
        out[..., :3] = torch.clamp(raw[..., :3], -1.0, 1.0)
        out[...,  3] = torch.clamp(raw[...,  3],  0.0, 1.0)
        return out


class GATMAPPOCritic(nn.Module):
    """
    Transfer learning Critic — 4-yönlü split giriş.

    Global obs yapısı (156D = 2×78):
      [0:100]   fc1_old      — eski base global obs (checkpoint'ten)
      [100:136] fc1_new      — eski ext 2×18D (checkpoint'ten, başta dondurulmuş)
      [136:148] fc1_intent_c — yeni OM intent 2×6D (sıfır init)
      [148:156] fc1_role_c   — yeni rol ataması 2×4D (sıfır init)
    """

    def __init__(self, old_global_dim: int, ext_dim: int,
                 intent_dim_c: int, role_dim_c: int, hidden: int = 256):
        super().__init__()
        self.old_global_dim = old_global_dim
        self._s0 = old_global_dim
        self._s1 = old_global_dim + ext_dim
        self._s2 = old_global_dim + ext_dim + intent_dim_c

        self.fc1_old      = nn.Linear(old_global_dim, hidden, bias=True)
        self.fc1_new      = nn.Linear(ext_dim,        hidden, bias=False)
        self.fc1_intent_c = nn.Linear(intent_dim_c,   hidden, bias=False)
        self.fc1_role_c   = nn.Linear(role_dim_c,     hidden, bias=False)
        self.fc2          = nn.Linear(hidden, hidden)
        self.out          = nn.Linear(hidden, 1)

        self._init_new_weights()

    def _init_new_weights(self):
        nn.init.zeros_(self.fc1_new.weight)
        nn.init.zeros_(self.fc1_intent_c.weight)
        nn.init.zeros_(self.fc1_role_c.weight)
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.zeros_(self.fc2.bias)
        nn.init.orthogonal_(self.out.weight, gain=0.01)
        nn.init.zeros_(self.out.bias)

    def freeze_old(self):
        self.fc1_old.requires_grad_(False)
        self.fc1_new.requires_grad_(False)

    def unfreeze_old(self):
        self.fc1_old.requires_grad_(True)
        self.fc1_new.requires_grad_(True)

    def forward(self, global_obs: "torch.Tensor") -> "torch.Tensor":
        feat = F.relu(
            self.fc1_old(global_obs[..., :self._s0]) +
            self.fc1_new(global_obs[..., self._s0:self._s1]) +
            self.fc1_intent_c(global_obs[..., self._s1:self._s2]) +
            self.fc1_role_c(global_obs[..., self._s2:])
        )
        feat = F.relu(self.fc2(feat))
        return self.out(feat).squeeze(-1)

    @torch.no_grad()
    def act(self, global_obs: "torch.Tensor") -> float:
        return float(self.forward(global_obs).item())


class MAPPOTrainer:
    """
    MAPPO eğitim motoru.

    Parametreler config.yaml'dan okunur.
    """

    def __init__(self, config: dict, device: str = "auto", mode: str = "mappo_gat_om"):
        self.config = config
        tr = config["training"]

        # Cihaz
        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        # Curriculum
        self.curriculum = CurriculumManager(config)

        # Ortam — Faz 1 ile başlat (1v1)
        self.env = DogfightEnv(config)
        self.env.seed(int(tr.get("seed", 42)))
        self.env.set_curriculum_phase(1)

        # Eğitilen ajan ID'leri (Blue takımı, Faz 1'de sadece blue_0)
        self.train_ids = list(self.env.blue_ids)
        self.n_agents  = len(self.train_ids)

        # Ajan indexi
        self.agent_idx = {aid: i for i, aid in enumerate(self.train_ids)}

        # Observation boyutları
        # obs_dim: env tarafından her zaman max (2v2=50) olarak hesaplanır
        self.obs_dim        = self.env.obs_dim  # 50
        # global_obs_dim: her zaman max ajan sayısı × obs_dim (100)
        # Faz 1/2'de blue_1 slotu sıfırla doldurulur → ağ boyutu değişmez
        self.global_obs_dim = self.obs_dim * self.env._max_n_per_team  # 100
        self.action_dim     = self.env.action_dim  # 5

        # Hyperparametreler
        self.gamma         = float(tr.get("gamma",          0.99))
        self.gae_lambda    = float(tr.get("gae_lambda",     0.95))
        self.clip_eps      = float(tr.get("clip_epsilon",   0.2))
        self.entropy_coeff  = float(tr.get("entropy_coeff",  0.15))
        self.mean_pen_coeff = float(tr.get("mean_penalty_coeff", 0.01))
        self.vf_coeff       = float(tr.get("value_loss_coeff", 0.5))
        self.max_grad_norm = float(tr.get("max_grad_norm",  0.5))
        self.n_steps       = int(tr.get("n_steps",          128))
        self.n_epochs      = int(tr.get("n_epochs",         4))
        self.minibatch     = int(tr.get("minibatch_size",   64))
        self.total_steps   = int(tr.get("total_timesteps",  10_000_000))
        self.lr_actor      = float(tr.get("lr_actor",       3e-4))
        self.lr_critic     = float(tr.get("lr_critic",      3e-4))

        hidden = int(tr.get("hidden_dim", 256))

        # GAT iletişim modu
        comm = config.get("communication", {})
        self.mode      = mode
        self.gat_mode  = mode in ("mappo_gat", "mappo_gat_om")
        self.om_mode   = mode == "mappo_gat_om"
        self.base_obs_dim  = self.obs_dim   # Faz-1 baseline: 50
        self._old_unfrozen = False
        self._freeze_steps = int(comm.get("freeze_steps", 500_000))

        if self.gat_mode:
            from models.gat_comm import GATComm
            _role_dim       = 4 if self.om_mode else 0
            _gat_msg        = int(comm.get("msg_dim", 16))
            _opp_intent_dim = 6 if self.om_mode else 0   # 2 düşman × 3D intent
            _ext_dim        = _gat_msg + 2   # 16D GAT + 2D dummy = 18D
            self.obs_dim        = self.base_obs_dim + _ext_dim + _opp_intent_dim + _role_dim
            # mappo_gat: 68D  |  mappo_gat_om: 78D
            self.global_obs_dim = self.obs_dim * self.env._max_n_per_team
            self._gat_node_dim  = int(comm.get("node_dim", 17))
            self._gat_wez_range = float(config.get("weapons", {})
                                        .get("wez_range_max", 8000.0))

            self.gat_comm = GATComm(
                node_dim = self._gat_node_dim,
                edge_dim = int(comm.get("edge_dim", 3)),
                n_heads  = int(comm.get("n_heads",  4)),
                msg_dim  = _gat_msg,
            ).to(self.device)

            self.actor = GATMAPPOActor(
                old_obs_dim  = self.base_obs_dim,
                ext_dim      = _ext_dim,
                intent_dim   = _opp_intent_dim,
                role_dim     = _role_dim,
                action_dim   = self.action_dim,
                hidden       = hidden,
            ).to(self.device)
            _n = self.env._max_n_per_team
            self.critic = GATMAPPOCritic(
                old_global_dim = self.base_obs_dim * _n,
                ext_dim        = _ext_dim * _n,
                intent_dim_c   = _opp_intent_dim * _n,
                role_dim_c     = _role_dim * _n,
                hidden         = hidden,
            ).to(self.device)

            # OM bileşenler (sadece om_mode)
            if self.om_mode:
                self.cent_om    = CentralizedOpponentModel().to(self.device)
                self.cent_role  = CentralizedRoleAssigner().to(self.device)
                self.enemy_hist = EnemyHistoryBuffer()
                self.om_replay  = OMReplayBuffer(capacity=5_000)
                self.opt_om     = optim.Adam(self.cent_om.parameters(), lr=1e-3)
                self.opt_role   = optim.Adam(self.cent_role.parameters(), lr=3e-4)
                self._intent_acc:    list = []
                self._role_acc:      list = []
                self._role_inp_buf:  list = []
                self._role_pair_buf: list = []          # REINFORCE için pair indeksleri
                self._return_history = deque(maxlen=200)  # REINFORCE baseline

            # opt_actor: gat_comm + fc1_new + fc1_intent + fc1_role + tail (fc1_old başta dondurulmuş)
            _actor_params = (list(self.gat_comm.parameters()) +
                             [p for n, p in self.actor.named_parameters()
                              if 'fc1_old' not in n])
            _critic_tail  = [p for n, p in self.critic.named_parameters()
                             if 'fc1_old' not in n]
            self.opt_actor  = optim.Adam(_actor_params, lr=self.lr_actor)
            self.opt_critic = optim.Adam(_critic_tail,  lr=self.lr_critic)
            self.actor.freeze_old()
            self.critic.freeze_old()
        else:
            self.gat_comm   = None
            self.actor  = MAPPOActor(self.obs_dim, self.action_dim,
                                      hidden=hidden).to(self.device)
            self.critic = MAPPOCritic(self.global_obs_dim,
                                       hidden=hidden).to(self.device)
            self.opt_actor  = optim.Adam(self.actor.parameters(),  lr=self.lr_actor)
            self.opt_critic = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        # Heuristic rakip — tüm ajan ID'lerini kapsayacak şekilde (max=2v2)
        # Faz değişince _rebuild_opp_policy çağrılır
        self._rebuild_opp_policy()

        # Fictitious self-play opponent pool
        pool_cfg = config.get("opponent_pool", {})
        self.pool_start_step      = int(pool_cfg.get("start_step",            500_000))
        self.pool_update_interval = int(pool_cfg.get("pool_update_interval",  200))
        self._rebuild_pool(pool_cfg)

        # Buffer — her zaman max n_agents (2) boyutunda
        max_n = self.env._max_n_per_team
        self.buffer = RolloutBuffer(
            self.n_steps, max_n,
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
        self._ep_rewards     = []
        self._ep_wins        = []   # 1=win, 0=diğer
        self._ep_losses      = []   # 1=loss, 0=diğer
        self._ep_draws       = []   # 1=draw, 0=diğer
        self._ep_lengths     = []
        self._ep_reasons     = []   # "win" | "loss" | "draw" | "timeout"
        self._ep_kills_blue  = []   # gerçek Red kill sayısı per episode
        self._ep_second_kill = []   # 1=kills>=2, 0=diğer
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

        print(f"\n[MAPPO] Eğitim başlıyor — toplam {self.total_steps:,} adım\n", flush=True)
        t_start = time.time()
        self._train_start_step = self.global_step  # epsilon decay için referans

        while self.global_step < self.total_steps:

            # ── Rollout toplama ────────────────────────────────────────
            self.buffer.reset()

            for _ in range(self.n_steps):

                # GAT: 50D obs → 68D/78D (GAT mesajı + rol embedding)
                # Pool ve env hâlâ 50D obs_dict kullanır
                if self.gat_mode:
                    obs_ext = self._extend_obs_gat(obs_dict)
                else:
                    obs_ext = obs_dict

                # Eğitilen ajanlar için aksiyon (genişletilmiş obs ile)
                actions_train, log_probs_train, values = \
                    self._collect_train_actions(obs_ext)

                # Opponent aksiyonları: pool 50D obs kullanır
                state_dict  = self.env.get_all_states()
                actions_opp = self.pool.act(obs_dict, state_dict)

                # Tüm aksiyonları birleştir
                action_dict = {**actions_train, **actions_opp}

                # Global obs (critic girişi) — genişletilmiş obs
                global_obs = self._build_global_obs(obs_ext)

                # om_mode: rol obs_ext[74:78]'den oku (yeni layout: base50|ext18|intent6|role4)
                role_support_probs = None
                role_vecs          = None
                if self.om_mode:
                    role_support_probs = {}
                    role_vecs          = {}
                    for aid in self.train_ids:
                        if aid not in obs_ext:
                            continue
                        role_np = obs_ext[aid][74:78]   # [74:78] = role(4)
                        role_support_probs[aid] = float(role_np[3])
                        role_vecs[aid]          = role_np

                # Adım
                next_obs, rew_dict, done_dict, info_dict = \
                    self.env.step(action_dict, role_support_probs=role_support_probs,
                                  role_vecs=role_vecs)

                # Buffer'a ekle — genişletilmiş obs saklanır (phase2: 76D, diğer: 50D)
                max_n      = self.env._max_n_per_team
                all_blue   = [f"blue_{i}" for i in range(max_n)]
                pad_obs    = {aid: (obs_ext[aid] if aid in obs_ext
                                    else np.zeros(self.obs_dim, dtype=np.float32))
                              for aid in all_blue}
                pad_act    = {aid: (actions_train[aid] if aid in actions_train
                                    else np.zeros(self.action_dim, dtype=np.float32))
                              for aid in all_blue}
                pad_lp     = {aid: (log_probs_train[aid] if aid in log_probs_train
                                    else 0.0)
                              for aid in all_blue}
                pad_rew    = {aid: (rew_dict[aid] if aid in rew_dict else 0.0)
                              for aid in all_blue}
                pad_values = np.zeros(max_n, dtype=np.float32)
                pad_values[:len(values)] = values

                self.buffer.add(
                    obs        = pad_obs,
                    actions    = pad_act,
                    log_probs  = pad_lp,
                    rewards    = pad_rew,
                    dones      = done_dict,
                    values     = pad_values,
                    global_obs = global_obs,
                    agent_ids  = all_blue,
                )

                # İstatistik
                for aid in self.train_ids:
                    ep_reward[aid] += rew_dict[aid]
                ep_steps += 1
                self.global_step += len(self.train_ids)

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
                    # REINFORCE: cent_role gradient (her episode sonunda)
                    if self.om_mode:
                        self._update_role_reinforce(float(mean_rew))
                    self._ep_rewards.append(mean_rew)
                    self._ep_wins.append(is_win)
                    self._ep_losses.append(is_loss)
                    self._ep_draws.append(is_draw)
                    self._ep_lengths.append(ep_steps)
                    self._ep_reasons.append(reason)
                    self.episode_count += 1

                    # Pool'a episode sonucunu bildir (adaptif seçim için)
                    self.pool.record_outcome(is_win)

                    # Gerçek kill sayısı: ölü Red sayısı (STATE_ALIVE < 0.5)
                    ep_kills = sum(
                        1 for rid in self.env.red_ids
                        if self.env._states.get(rid, np.zeros(1))[STATE_ALIVE] < 0.5
                    )
                    self._ep_kills_blue.append(ep_kills)
                    self._ep_second_kill.append(1 if ep_kills >= 2 else 0)
                    self.curriculum.record_episode(ep_kills, is_win)

                    # Faz 1.5: mesafe güncellemesi sonrası env'e sync et
                    if self.curriculum.phase == 2:
                        self.env.set_dynamic_spawn_dist(
                            self.curriculum.current_spawn_dist
                        )

                    if self.episode_count % self.log_interval == 0:
                        self._log_progress(t_start)
                        print(self.pool.log_win_rate_distribution())
                        print(f"[Curriculum] {self.curriculum.status_str()}")

                    if self.episode_count % self.ckpt_interval == 0:
                        self._save_checkpoint()

                    # Phase-2: eski ağırlıkları çöz (aşama 1 → 2)
                    if (self.gat_mode and not self._old_unfrozen
                            and self.global_step >= self._freeze_steps):
                        self._unfreeze_old_weights()

                    if self.episode_count % self.pool_update_interval == 0:
                        snap = self._save_pool_snapshot()
                        self.pool.add_checkpoint(snap)

                    # Curriculum geçiş kontrolü
                    if self.curriculum.check_transition():
                        self._save_checkpoint()
                        self._advance_curriculum()

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
            self._update_om()

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

    def _role_epsilon(self) -> float:
        """
        Epsilon-greedy rol keşfi için lineer decay.
        Bu run'ın başından itibaren sayar (_train_start_step).
        """
        if not self.om_mode:
            return 0.0
        cfg       = self.config.get("role_exploration", {})
        eps_start = float(cfg.get("role_epsilon_start", 0.0))
        eps_end   = float(cfg.get("role_epsilon_end",   0.0))
        eps_steps = int(cfg.get("role_epsilon_steps",   10_000_000))
        if eps_start <= 0.0:
            return 0.0
        steps_done = self.global_step - getattr(self, "_train_start_step", self.global_step)
        t = min(1.0, steps_done / max(eps_steps, 1))
        return float(eps_start + (eps_end - eps_start) * t)

    def _update_om(self) -> float:
        """CentralizedOpponentModel supervised loss (sadece om_mode)."""
        if not self.om_mode:
            return 0.0
        if len(self.om_replay) < 32:
            return 0.0
        batch     = min(len(self.om_replay), 128)
        hist_np, labels_np = self.om_replay.sample(batch)
        hist_t    = torch.FloatTensor(hist_np).to(self.device)
        label_t   = torch.LongTensor(labels_np).to(self.device)
        om_loss   = self.cent_om.supervised_loss(hist_t, label_t)
        self.opt_om.zero_grad()
        om_loss.backward()
        nn.utils.clip_grad_norm_(self.cent_om.parameters(), 0.5)
        self.opt_om.step()
        return float(om_loss.item())

    def _update_role_reinforce(self, episode_return: float) -> None:
        """
        Episode-level REINFORCE ile cent_role eğitimi.
        Rol atamalarının log-olasılığı × normalize_return ile gradient geçirir.
        Her episode sonunda çağrılır (_update_om'dan bağımsız).
        """
        if not self.om_mode:
            return
        if len(self._role_pair_buf) < 1:
            return

        self._return_history.append(episode_return)
        if len(self._return_history) < 10:
            self._role_pair_buf.clear()
            self._role_inp_buf.clear()
            return

        ret_mean = float(np.mean(self._return_history))
        ret_std  = float(np.std(self._return_history)) + 1e-8
        ret_norm = float(np.clip((episode_return - ret_mean) / ret_std, -3.0, 3.0))

        # Cent_role MLP'sini geçmişe tekrar koştur (gradient AKIYOR)
        x_np    = np.array(self._role_inp_buf, dtype=np.float32)   # [T, inp_dim]
        x_t     = torch.FloatTensor(x_np).to(self.device)
        logits  = self.cent_role.mlp(x_t)                          # [T, 12]
        log_p   = F.log_softmax(logits, dim=-1)                    # [T, 12]

        pair_t  = torch.LongTensor(self._role_pair_buf).to(self.device)  # [T]
        T       = len(pair_t)
        assigned_lp = log_p[torch.arange(T, device=self.device), pair_t]  # [T]

        # Çeşitlilik entropisi bonusu
        probs   = F.softmax(logits, dim=-1)
        entropy = -(probs * log_p).sum(dim=-1).mean()

        # REINFORCE + entropy
        role_loss = -(ret_norm * assigned_lp.mean()) - 0.05 * entropy

        self.opt_role.zero_grad()
        role_loss.backward()
        nn.utils.clip_grad_norm_(self.cent_role.parameters(), 0.5)
        self.opt_role.step()

        self._role_pair_buf.clear()
        self._role_inp_buf.clear()

    def _update(self):
        """GAE + PPO clip loss + value loss + entropy."""
        # Bootstrap value — gat_mode'da obs 68D/78D'ye genişletilmeli
        obs_dict = self.env._build_obs_dict()
        if self.gat_mode:
            obs_dict = self._extend_obs_gat(obs_dict)
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

        # Tensor dönüşüm — buffer her zaman max_n (2) ajan için tutulur
        T = self.n_steps
        A = self.env._max_n_per_team  # 2 (sabit)
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

                # Mean-head L2 penalty — tanh saturasyonunu önler (raw değerleri ±3 içinde tutar)
                mean_penalty = self.mean_pen_coeff * ctrl_dist.mean.pow(2).mean()

                # Value loss
                value_pred = self.critic(mb_gobs).squeeze(-1)
                value_loss = nn.functional.mse_loss(value_pred, mb_ret)

                # Toplam kayıp
                loss = (actor_loss
                        + self.vf_coeff * value_loss
                        - self.entropy_coeff * entropy
                        + mean_penalty)

                self.opt_actor.zero_grad()
                self.opt_critic.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(),
                                          self.max_grad_norm)
                if self.gat_comm is not None:
                    nn.utils.clip_grad_norm_(self.gat_comm.parameters(),
                                              self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(),
                                          self.max_grad_norm)
                self.opt_actor.step()
                self.opt_critic.step()

        self._update_count += 1

    # -----------------------------------------------------------------------
    # Yardımcı Metodlar
    # -----------------------------------------------------------------------

    def _rebuild_opp_policy(self):
        """Mevcut faz ajan listesine göre heuristic policy yeniden oluşturur."""
        team_map = {aid: ("blue" if "blue" in aid else "red")
                    for aid in self.env.agent_ids}
        self.opp_policy = MultiHeuristicPolicy(
            self.config, self.env.agent_ids, team_map
        )

    def _rebuild_pool(self, pool_cfg: dict = None):
        """Mevcut faz ajan listesine göre OpponentPool yeniden oluşturur."""
        if pool_cfg is None:
            pool_cfg = self.config.get("opponent_pool", {})
        self.pool = OpponentPool(
            config        = self.config,
            red_ids       = self.env.red_ids,
            obs_dim       = self.obs_dim,
            action_dim    = self.action_dim,
            device        = self.device,
            fallback      = self.opp_policy,
            max_pool_size = int(pool_cfg.get("max_pool_size", 20)),
        )

    def _advance_curriculum(self):
        """Bir sonraki curriculum fazına geç; env ve policy'leri güncelle."""
        old_phase = self.curriculum.phase
        self.curriculum.advance()
        new_phase = self.curriculum.phase

        self.env.set_curriculum_phase(new_phase)
        self.train_ids = list(self.env.blue_ids)
        self.n_agents  = len(self.train_ids)
        self.agent_idx = {aid: i for i, aid in enumerate(self.train_ids)}

        # Faz 1.5: env'e başlangıç spawn mesafesini bildir
        if new_phase == 2:
            self.env.set_dynamic_spawn_dist(self.curriculum.current_spawn_dist)

        self._rebuild_opp_policy()
        self._rebuild_pool()

        # _blue_heuristic yeniden oluşturulmalı (yeni ajan listesi)
        if hasattr(self, '_blue_heuristic'):
            del self._blue_heuristic

        print(f"\n{'='*60}")
        print(f"[Curriculum] FAZ GECISI: {old_phase} -> {new_phase}")
        print(f"[Curriculum] {self.curriculum.PHASE_NAMES.get(new_phase)}")
        print(f"[Curriculum] n_per_team={self.env.n_per_team}, "
              f"train_ids={self.train_ids}")
        print(f"{'='*60}\n")

    # -----------------------------------------------------------------------
    # Phase-2 Transfer Learning Yardımcıları
    # -----------------------------------------------------------------------

    def _unfreeze_old_weights(self):
        """
        Transfer learning aşama 1 → 2: fc1_old parametrelerini eğitime aç.
        fc1_new ve gat_comm zaten opt_actor'de; sadece fc1_old eklenir.
        """
        self.actor.unfreeze_old()
        self.critic.unfreeze_old()
        self.opt_actor.add_param_group(
            {'params': list(self.actor.fc1_old.parameters()),
             'lr': self.lr_actor}
        )
        self.opt_critic.add_param_group(
            {'params': list(self.critic.fc1_old.parameters()),
             'lr': self.lr_critic}
        )
        self._old_unfrozen = True
        print(f"\n[Phase2] Adım {self.global_step:,}: "
              f"eski ağırlıklar çözüldü — tam fine-tune başladı\n", flush=True)

    def _build_gat_edge_feats(self, blue_ids: list) -> np.ndarray:
        """
        Mavi takım ajanları arasında kenar özellik matrisi hesaplar.

        Returns
        -------
        np.ndarray — (n_agents, n_agents, 3)
            [:, i, j, :] = [distance_norm, bearing_norm, threat_j]
        """
        N = len(blue_ids)
        edge   = np.zeros((N, N, 3), dtype=np.float32)
        states = self.env.get_all_states()

        for i, aid_i in enumerate(blue_ids):
            for j, aid_j in enumerate(blue_ids):
                if i == j:
                    continue
                s_i = states.get(aid_i)
                s_j = states.get(aid_j)
                if s_i is None or s_j is None:
                    continue

                pos_i = s_i[[STATE_X, STATE_Y, STATE_H]]
                pos_j = s_j[[STATE_X, STATE_Y, STATE_H]]
                dist  = distance_3d(pos_i, pos_j)
                bear  = bearing_angle(pos_i, pos_j)

                # j'nin en yakın kırmızı ajana olan tehdit skoru
                ts = 0.0
                for eid in self.env.red_ids:
                    es = states.get(eid)
                    if es is not None and es[STATE_ALIVE] > 0.5:
                        d  = distance_3d(pos_j, es[[STATE_X, STATE_Y, STATE_H]])
                        ts = max(ts, float(np.clip(
                            1.0 - d / (self._gat_wez_range + 1e-9), 0.0, 1.0
                        )))

                edge[i, j] = [
                    float(np.clip(dist / (self.env.map_size + 1e-9), 0.0, 1.0)),
                    float(np.clip(wrap_to_pi(bear) / np.pi, -1.0, 1.0)),
                    ts,
                ]
        return edge

    def _extend_obs_gat(self, obs_dict: dict) -> dict:
        """
        gat_mode=True obs genişletici.
        om_mode=True : 50D → 78D (base+ext18+intent6+role4)
        om_mode=False: 50D → 68D (base+ext18)

        Centralized OM: tek bir 960D geçmiş buffer'dan 6D intent üretir.
        Centralized RoleAssigner: intent+team_state → çakışmasız rol ataması.
        """
        blue_ids = [f"blue_{i}" for i in range(self.env._max_n_per_team)]
        states   = self.env.get_all_states()

        # ── GAT mesajları (her zaman) ──────────────────────────────────────
        ego_list   = []
        alive_list = []
        for aid in blue_ids:
            obs = obs_dict.get(aid, np.zeros(self.base_obs_dim, dtype=np.float32))
            ego_list.append(obs[:self._gat_node_dim])
            s = states.get(aid)
            alive_list.append(float(s[STATE_ALIVE]) if s is not None else 0.0)
        edge_feats = self._build_gat_edge_feats(blue_ids)
        messages   = self.gat_comm.compute_messages(
            ego_list, edge_feats, alive_list, self.device
        )

        if not self.om_mode:
            # 68D: base(50) + 2D_dummy + 16D_GAT
            extended = {}
            for i, aid in enumerate(blue_ids):
                base  = obs_dict.get(aid, np.zeros(self.base_obs_dim, dtype=np.float32))
                ext18 = np.concatenate([np.zeros(2, dtype=np.float32), messages[i]])
                extended[aid] = np.concatenate([base, ext18])  # 68D
            return extended

        # ── Centralized OM: 960D geçmiş → 6D intent ───────────────────────
        obs_arr = np.stack([
            obs_dict.get(aid, np.zeros(self.base_obs_dim, dtype=np.float32))
            for aid in blue_ids
        ], axis=0)  # (2, 50)
        hist_960 = self.enemy_hist.update(obs_arr)

        with torch.no_grad():
            hist_t    = torch.from_numpy(hist_960).unsqueeze(0).to(self.device)
            intent_np = self.cent_om.intent_flat(hist_t).squeeze(0).cpu().numpy()  # (6,)

        # OM supervised replay doldur
        if np.any(hist_960 != 0):
            valid_pairs = []
            for aid in blue_ids:
                bs = states.get(aid)
                if bs is not None:
                    valid_pairs.append((bs, float(bs[STATE_ALIVE])))
            label0 = INTENT_DEFENSIVE
            label1 = INTENT_DEFENSIVE
            if len(self.env.red_ids) > 0:
                rs0 = states.get(self.env.red_ids[0])
                if rs0 is not None:
                    label0 = get_om_label(rs0, valid_pairs)
            if len(self.env.red_ids) > 1:
                rs1 = states.get(self.env.red_ids[1])
                if rs1 is not None:
                    label1 = get_om_label(rs1, valid_pairs)
            self.om_replay.add(hist_960, label0, label1)

        # ── Centralized Role Assignment ────────────────────────────────────
        ts_np     = build_team_state(
            [states.get(f"blue_{i}") for i in range(self.env._max_n_per_team)]
        )
        x_role_np = np.concatenate([intent_np, ts_np]).astype(np.float32)
        with torch.no_grad():
            x_role  = torch.from_numpy(x_role_np).unsqueeze(0).to(self.device)
            role_0, role_1 = self.cent_role(x_role, hard=True, tau=0.5)
        role_0_np = role_0.squeeze(0).cpu().numpy()  # (4,) one-hot
        role_1_np = role_1.squeeze(0).cpu().numpy()

        # REINFORCE: seçilen pair indeksini kaydet
        r0_idx   = int(role_0_np.argmax())
        r1_idx   = int(role_1_np.argmax())
        pair_idx = ROLE_PAIRS.index((r0_idx, r1_idx))
        self._role_pair_buf.append(pair_idx)

        role_vecs = {blue_ids[0]: role_0_np, blue_ids[1]: role_1_np}

        # Logging buffers
        self._intent_acc.append(intent_np.copy())
        self._role_acc.append(role_0_np.copy())
        self._role_inp_buf.append(x_role_np.copy())

        # ── 78D obs: base(50) + 2D_dummy+GAT16(18) + intent(6) + role(4) ──
        extended = {}
        for i, aid in enumerate(blue_ids):
            base  = obs_dict.get(aid, np.zeros(self.base_obs_dim, dtype=np.float32))
            ext18 = np.concatenate([np.zeros(2, dtype=np.float32), messages[i]])
            extended[aid] = np.concatenate([base, ext18, intent_np, role_vecs[aid]])  # 78D
        return extended

    def _reset_episode(self) -> dict:
        self.pool.reset()
        if self.gat_mode and self.om_mode:
            self.enemy_hist.reset()
        return self.env.reset()

    def _save_pool_snapshot(self) -> str:
        """Actor ağırlıklarını pool snapshot olarak kaydet (optimizer olmadan)."""
        path = self.ckpt_dir / f"pool_actor_ep{self.episode_count}.pt"
        snap = {
            "episode":     self.episode_count,
            "global_step": self.global_step,
            "actor":       self.actor.state_dict(),
            "config":      self.config,
        }
        if self.gat_mode and self.gat_comm is not None:
            snap["gat_comm"] = self.gat_comm.state_dict()
            snap["mode"]     = self.mode
        torch.save(snap, path)
        return str(path)

    def _build_global_obs(self, obs_dict: dict) -> np.ndarray:
        """
        Eğitilen ajanların obs'unu birleştir → (global_obs_dim,).

        Faz 1/2'de blue_1 eksik olduğundan zeros ile doldurulur
        → global_obs_dim = obs_dim × max_n_per_team (100) sabit kalır.
        """
        max_n = self.env._max_n_per_team
        all_blue = [f"blue_{i}" for i in range(max_n)]
        parts = []
        for aid in all_blue:
            if aid in obs_dict:
                parts.append(obs_dict[aid])
            else:
                parts.append(np.zeros(self.obs_dim, dtype=np.float32))
        return np.concatenate(parts, axis=0)

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

        # Gerçek kill metrikleri
        kill_per_ep     = float(np.mean(self._ep_kills_blue[-window:]))   if self._ep_kills_blue  else 0.0
        second_kill_rate = float(np.mean(self._ep_second_kill[-window:])) if self._ep_second_kill else 0.0

        print(
            f"[Ep {self.episode_count:>6}] "
            f"step={self.global_step:>9,} | "
            f"rew={mean_rew:>7.2f} | "
            f"W={win_rate:.2f} L={loss_rate:.2f} D={draw_rate:.2f} | "
            f"kills={kill_per_ep:.2f} 2nd={second_kill_rate:.2f} | "
            f"len={mean_len:>5.0f} | "
            f"{steps_sec:>5.0f}sps",
            flush=True
        )
        print(
            f"{'':>10}bitiş: "
            f"win={r_counts['win']:>3} "
            f"loss={r_counts['loss']:>3} "
            f"draw={r_counts['draw']:>3} "
            f"timeout={r_counts['timeout']:>3}",
            flush=True
        )

        # Intent / Role dağılımı (sadece phase2)
        intent_vals = [0.0] * 6   # [agg0, def0, eva0, agg1, def1, eva1]
        role_vals   = [0.25] * 4  # [sniper, pursuit, defensive, support]
        if self.om_mode:
            if self._intent_acc:
                im = np.mean(self._intent_acc, axis=0)  # (6,)
                intent_vals = [round(float(v), 4) for v in im]
                self._intent_acc.clear()
                print(
                    f"{'':>10}intent: "
                    f"e0[agg={intent_vals[0]:.2f} def={intent_vals[1]:.2f} eva={intent_vals[2]:.2f}] "
                    f"e1[agg={intent_vals[3]:.2f} def={intent_vals[4]:.2f} eva={intent_vals[5]:.2f}]",
                    flush=True
                )
            if self._role_acc:
                rm = np.mean(self._role_acc, axis=0)  # (4,)
                role_vals = [round(float(v), 4) for v in rm]
                self._role_acc.clear()
                print(
                    f"{'':>10}role:   "
                    f"sniper={role_vals[0]:.2f} pursuit={role_vals[1]:.2f} "
                    f"defensive={role_vals[2]:.2f} support={role_vals[3]:.2f}",
                    flush=True
                )

        # CSV
        csv_path = self.log_dir / "train_log.csv"
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["episode", "global_step", "mean_reward",
                             "win_rate", "loss_rate", "draw_rate",
                             "kills_blue", "second_kill_rate",
                             "mean_ep_len", "updates",
                             "n_win", "n_loss", "n_draw", "n_timeout",
                             "intent_agg0", "intent_def0", "intent_eva0",
                             "intent_agg1", "intent_def1", "intent_eva1",
                             "role_sniper", "role_pursuit",
                             "role_defensive", "role_support"])
            w.writerow([
                self.episode_count, self.global_step,
                round(mean_rew, 4), round(win_rate, 4),
                round(loss_rate, 4), round(draw_rate, 4),
                round(kill_per_ep, 4), round(second_kill_rate, 4),
                round(mean_len, 1), self._update_count,
                r_counts["win"], r_counts["loss"],
                r_counts["draw"], r_counts["timeout"],
                *intent_vals, *role_vals,
            ])

    def _save_checkpoint(self, final: bool = False):
        """Actor + Critic ağırlıklarını kaydet."""
        tag    = "final" if final else f"ep{self.episode_count}"
        prefix = self.mode
        path   = self.ckpt_dir / f"{prefix}_{tag}.pt"
        ckpt = {
            "episode":      self.episode_count,
            "global_step":  self.global_step,
            "actor":        self.actor.state_dict(),
            "critic":       self.critic.state_dict(),
            "opt_actor":    self.opt_actor.state_dict(),
            "opt_critic":   self.opt_critic.state_dict(),
            "config":       self.config,
            "mode":         self.mode,
        }
        if self.gat_mode and self.gat_comm is not None:
            ckpt["gat_comm"]     = self.gat_comm.state_dict()
            ckpt["old_unfrozen"] = self._old_unfrozen
            if self.om_mode:
                ckpt["cent_om"]   = self.cent_om.state_dict()
                ckpt["cent_role"] = self.cent_role.state_dict()
                ckpt["opt_role"]  = self.opt_role.state_dict()
        torch.save(ckpt, path)
        print(f"[MAPPO] Checkpoint kaydedildi: {path}")

    def _reset_mismatched_opt_states(self, optimizer, model):
        """fc1_new boyutu değiştiyse (ör. 24D→26D) Adam momentum buffer'larını sıfırla."""
        for name, param in model.named_parameters():
            if 'fc1_new' not in name:
                continue
            if param not in optimizer.state:
                continue
            stored = optimizer.state[param]
            mismatch = any(
                isinstance(v, torch.Tensor) and v.shape != param.shape
                for v in stored.values()
            )
            if mismatch:
                del optimizer.state[param]
                print(f"[MAPPO] {name} optimizer state sıfırlandı (boyut uyuşmazlığı)")

    def _load_state_dict_with_fc1new_expand(self, model, ckpt_state: dict, label: str):
        """
        fc1_new boyutu eski checkpoint ile yeni model arasında uyuşmazsa:
        ilk N sütunu kopyala, kalan sütunları sıfır ile doldur.
        Uyuşuyorsa normal load_state_dict.
        """
        model_state = model.state_dict()
        key = "fc1_new.weight"
        if key in ckpt_state and key in model_state:
            ckpt_w  = ckpt_state[key]   # (256, old_in)
            model_w = model_state[key]  # (256, new_in)
            if ckpt_w.shape != model_w.shape:
                old_in = ckpt_w.shape[1]
                new_in = model_w.shape[1]
                ckpt_state = dict(ckpt_state)  # shallow copy — orijinali değiştirme
                if old_in <= new_in:
                    # Genişletme: sütunları kopyala, kalan sıfır
                    expanded = torch.zeros_like(model_w)
                    expanded[:, :old_in] = ckpt_w
                    ckpt_state[key] = expanded
                    print(f"[MAPPO] {label} fc1_new genişletiliyor: {old_in}D → {new_in}D")
                else:
                    # Daralma: yeni split mimarisinde fc1_new sıfır init
                    ckpt_state[key] = torch.zeros_like(model_w)
                    print(f"[MAPPO] {label} fc1_new sıfırlanıyor: {old_in}D → {new_in}D (yeni split)")
        model.load_state_dict(ckpt_state, strict=False)

    def load_checkpoint(self, path: str):
        """Checkpoint yükle (eval veya devam eğitimi için)."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self._load_state_dict_with_fc1new_expand(self.actor,  ckpt["actor"],  "actor")
        self._load_state_dict_with_fc1new_expand(self.critic, ckpt["critic"], "critic")
        if self.gat_mode and "gat_comm" in ckpt:
            self.gat_comm.load_state_dict(ckpt["gat_comm"])
            if "cent_om" in ckpt:
                self.cent_om.load_state_dict(ckpt["cent_om"])
            if "cent_role" in ckpt:
                try:
                    self.cent_role.load_state_dict(ckpt["cent_role"])
                except RuntimeError:
                    print("[MAPPO] cent_role mimari değişti — sıfırdan başlıyor (joint pair)")
            if "opt_role" in ckpt:
                try:
                    self.opt_role.load_state_dict(ckpt["opt_role"])
                except ValueError:
                    print("[MAPPO] opt_role state uyuşmadı — sıfırlanıyor")
            old_unfrozen = ckpt.get("old_unfrozen", False)
            if old_unfrozen:
                # fc1_old daha önce açılmış: opt_actor'e grup ekle, sonra yükle
                self.actor.unfreeze_old()
                self.critic.unfreeze_old()
                self.opt_actor.add_param_group(
                    {'params': list(self.actor.fc1_old.parameters()),
                     'lr': self.lr_actor}
                )
                self.opt_critic.add_param_group(
                    {'params': list(self.critic.fc1_old.parameters()),
                     'lr': self.lr_critic}
                )
                self._old_unfrozen = True
            else:
                self.actor.freeze_old()
                self.critic.freeze_old()
        try:
            self.opt_actor.load_state_dict(ckpt["opt_actor"])
            self.opt_critic.load_state_dict(ckpt["opt_critic"])
            # fc1_new boyutu değiştiyse momentum buffer'ları sıfırla
            self._reset_mismatched_opt_states(self.opt_actor,  self.actor)
            self._reset_mismatched_opt_states(self.opt_critic, self.critic)
            print("[MAPPO] Optimizer state yüklendi (warm-start)")
        except ValueError:
            # Mimari değişince (yeni modül eklendi) optimizer group uyuşmaz.
            # Ağırlıklar yüklendi; optimizer sıfırdan devam eder.
            print("[MAPPO] opt_actor/critic state uyuşmadı — optimizer sıfırlanıyor "
                  "(beklenen: eski checkpoint'ten yeni mimari; sonraki resume'da warm-start)")
        self.episode_count = ckpt.get("episode",     0)
        self.global_step   = ckpt.get("global_step", 0)
        print(f"[MAPPO] Checkpoint yüklendi: {path} "
              f"(ep={self.episode_count}, step={self.global_step:,})")

    def load_phase2_checkpoint(self, path: str):
        """
        mappo_final.pt (Faz-1) ağırlıklarını Faz-2 ağına transfer eder.

        Actor  : net.0 → fc1_old(50→256) | net.2 → fc2 | head'ler
        Critic : net.0 → fc1_old(100→256)| net.2 → fc2 | net.4 → out
        GATComm: sıfırdan başlar (transfer yok)
        fc1_new: sıfır init (transfer yok)
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        old_actor  = ckpt["actor"]
        old_critic = ckpt["critic"]

        with torch.no_grad():
            # Actor transfer
            self.actor.fc1_old.weight.copy_(old_actor["net.0.weight"])
            self.actor.fc1_old.bias.copy_(old_actor["net.0.bias"])
            self.actor.fc2.weight.copy_(old_actor["net.2.weight"])
            self.actor.fc2.bias.copy_(old_actor["net.2.bias"])
            self.actor.mean_head.weight.copy_(old_actor["mean_head.weight"])
            self.actor.mean_head.bias.copy_(old_actor["mean_head.bias"])
            self.actor.log_std.copy_(old_actor["log_std"])
            self.actor.fire_head.weight.copy_(old_actor["fire_head.weight"])
            self.actor.fire_head.bias.copy_(old_actor["fire_head.bias"])

            # Critic transfer
            self.critic.fc1_old.weight.copy_(old_critic["net.0.weight"])
            self.critic.fc1_old.bias.copy_(old_critic["net.0.bias"])
            self.critic.fc2.weight.copy_(old_critic["net.2.weight"])
            self.critic.fc2.bias.copy_(old_critic["net.2.bias"])
            self.critic.out.weight.copy_(old_critic["net.4.weight"])
            self.critic.out.bias.copy_(old_critic["net.4.bias"])

        print(f"[Phase2] Transfer tamamlandı: {path}")
        print(f"[Phase2] Actor  — fc1_old(50→{self.actor.fc1_old.out_features})"
              f" + fc2 + heads kopyalandı | fc1_new sıfır | donduruldu")
        print(f"[Phase2] Critic — fc1_old(100→{self.critic.fc1_old.out_features})"
              f" + fc2 + out kopyalandı | fc1_new sıfır | donduruldu")
        gat_params = sum(p.numel() for p in self.gat_comm.parameters())
        print(f"[Phase2] GATComm sıfırdan başlıyor ({gat_params:,} param)")
        print(f"[Phase2] Dondurma süresi: {self._freeze_steps:,} adım")


# ===========================================================================
# Entry Point
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",  default="configs/config.yaml")
    p.add_argument("--seed",    type=int, default=None)
    p.add_argument("--device",  default="auto",
                   help="auto | cpu | cuda | cuda:0")
    p.add_argument("--mode",    default="mappo_gat_om",
                   choices=["mappo", "mappo_gat", "mappo_gat_om"],
                   help="Mimari: mappo | mappo_gat (GAT iletişim) | mappo_gat_om (GAT+OM+Rol)")
    p.add_argument("--resume",  default=None,
                   help="Aynı mimari checkpoint'ten devam")
    p.add_argument("--transfer", default=None,
                   help="Faz-1 MAPPO checkpoint'ten GAT mimarisine transfer")
    p.add_argument("--start-phase", type=int, default=None,
                   help="Curriculum fazını manuel olarak set et (1/2/3/4)")
    return p.parse_args()


def main():
    args = parse_args()

    config_path = PROJECT_ROOT / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if args.seed is not None:
        config["training"]["seed"] = args.seed

    trainer = MAPPOTrainer(config, device=args.device, mode=args.mode)

    if args.mode in ("mappo_gat", "mappo_gat_om"):
        if args.resume:
            trainer.load_checkpoint(args.resume)
        elif args.transfer:
            trainer.load_phase2_checkpoint(args.transfer)
    elif args.resume:
        trainer.load_checkpoint(args.resume)

    if args.start_phase is not None:
        old = trainer.curriculum.phase
        trainer.curriculum.phase = args.start_phase
        trainer.curriculum._ep_in_phase = 0
        trainer.curriculum._kill_history.clear()
        trainer.curriculum._win_history.clear()
        if args.start_phase == 2:
            trainer.curriculum.current_spawn_dist = \
                trainer.curriculum.phase15_dist_start
        trainer.env.set_curriculum_phase(args.start_phase)
        trainer.train_ids = list(trainer.env.blue_ids)
        trainer.n_agents  = len(trainer.train_ids)
        trainer.agent_idx = {aid: i for i, aid in enumerate(trainer.train_ids)}
        trainer._rebuild_opp_policy()
        trainer._rebuild_pool()
        if hasattr(trainer, '_blue_heuristic'):
            del trainer._blue_heuristic
        # Buffer'ı güncel n_agents ile yeniden oluştur
        max_n = trainer.env._max_n_per_team
        trainer.buffer = RolloutBuffer(
            trainer.n_steps, max_n,
            trainer.obs_dim, trainer.action_dim, trainer.global_obs_dim
        )
        print(f"[MAPPO] Curriculum fazi manuel set: {old} -> {args.start_phase} "
              f"({CurriculumManager.PHASE_NAMES.get(args.start_phase)})")
        print(f"[MAPPO] n_agents={trainer.n_agents}, buffer yeniden olusturuldu "
              f"(obs_dim={trainer.obs_dim}, global_obs={trainer.global_obs_dim})")

    trainer.train()


if __name__ == "__main__":
    main()