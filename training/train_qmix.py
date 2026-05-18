"""
train_qmix.py
=============
QMIX eğitim döngüsü — 4-fazlı curriculum + adaptif self-play pool.

4-Faz Curriculum:
  Faz-1: 1v1 WEZ-yakın  (500-2000m)  → kill/ep≥0.30 AND win≥0.40
  Faz-2: 1v1 kademeli   (4000→16km)  → kill/ep≥0.20, min 1200 ep
  Faz-3: 1v1 normal     (6000-12km)  → kill/ep≥0.15, min 300 ep
  Faz-4: 2v2 normal     (son faz)

Self-Play Pool:
  Her pool_update_interval episode'da Blue AgentQNetwork snapshot'ı alınır.
  Adaptif seçim: global win_rate < 0.2 → en zayıf, > 0.6 → en güçlü.

Kullanım:
    python -u -X utf8 training/train_qmix.py
    python -u -X utf8 training/train_qmix.py --test
    python -u -X utf8 training/train_qmix.py --resume checkpoints/qmix_ep1000.pt
    python -u -X utf8 training/train_qmix.py --resume checkpoints/qmix_ep5000.pt --start-phase 4
"""

import argparse
import copy
import csv
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.dogfight_env import DogfightEnv
from agents.heuristic_agent import MultiHeuristicPolicy
from models.qmix_net import ActionMapper, AgentQNetwork, QMixNet


# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

N_AGENTS       = 2
OBS_DIM        = 50
GLOBAL_OBS_DIM = N_AGENTS * OBS_DIM   # 100
ACTION_DIM     = 5
N_ACTIONS      = 162


# ===========================================================================
# Curriculum Manager
# ===========================================================================

class CurriculumManager:
    """
    4-fazlı curriculum yöneticisi (MAPPO ile aynı kriter ve fazlar).

    Dahili faz → env modu:
      1 = Faz-1   : 1v1 WEZ-yakın
      2 = Faz-1.5 : 1v1 kademeli mesafe
      3 = Faz-2   : 1v1 normal spawn
      4 = Faz-3   : 2v2 normal spawn
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
        self.phase1_win_thresh      = float(cur.get("phase1_win_threshold",  0.40))
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
        self.phase2_kill_thresh     = float(cur.get("phase2_kill_threshold", 0.15))
        self.phase2_min_ep          = int(cur.get("phase2_min_episodes",     300))

        self._kill_history: list    = []
        self._win_history:  list    = []
        self._ep_in_phase:  int     = 0
        self.current_spawn_dist     = self.phase15_dist_start

    def record_episode(self, kills: float, is_win: int):
        self._kill_history.append(float(kills))
        self._win_history.append(float(is_win))
        self._ep_in_phase += 1

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
            return (self._ep_in_phase >= self.phase2_min_ep
                    and recent_kill >= self.phase2_kill_thresh)
        return False

    def advance(self):
        self.phase += 1
        self._ep_in_phase = 0
        self._kill_history.clear()
        self._win_history.clear()
        if self.phase == 2:
            self.current_spawn_dist = self.phase15_dist_start

    def status_str(self) -> str:
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


# ===========================================================================
# QMIX Self-Play Pool
# ===========================================================================

class QMIXPool:
    """
    QMIX adaptif self-play pool.

    AgentQNetwork state_dict kopyaları saklar. Episode başında adaptif seçim
    ile bir snapshot yüklenir; Red ajanlar bu snapshot'ı Q-argmax + ActionMapper
    üzerinden kullanır. Pool boşken heuristic fallback devreye girer.
    """

    WIN_WINDOW    = 20
    WEAK_THRESH   = 0.2
    STRONG_THRESH = 0.6
    MIN_MATCHES   = 5

    def __init__(self, red_ids: list, agent_hidden: int, device, fallback,
                 action_mapper: ActionMapper, max_size: int = 20):
        self.red_ids      = red_ids
        self.agent_hidden = agent_hidden
        self.device       = device
        self.fallback     = fallback
        self.action_mapper= action_mapper
        self.max_size     = max_size

        self._pool:     list = []          # list of state_dict copies
        self._win_hist: list = []          # list of deque(maxlen=WIN_WINDOW)

        self._current_net: AgentQNetwork | None = None
        self._current_idx: int | None           = None
        self._use_fallback: bool                = True

    # -----------------------------------------------------------------------

    def add_checkpoint(self, agent_net: AgentQNetwork) -> None:
        sd = copy.deepcopy(agent_net.state_dict())
        self._pool.append(sd)
        self._win_hist.append(deque(maxlen=self.WIN_WINDOW))
        if len(self._pool) > self.max_size:
            self._pool.pop(0)
            self._win_hist.pop(0)

    @property
    def size(self) -> int:
        return len(self._pool)

    # -----------------------------------------------------------------------

    def reset(self) -> None:
        """Episode başı: checkpoint seç ve geçici ağa yükle."""
        self.fallback.reset()

        if len(self._pool) == 0:
            self._use_fallback = True
            self._current_net  = None
            self._current_idx  = None
            return

        idx = self._select_idx()
        self._current_idx  = idx
        self._use_fallback = False

        net = AgentQNetwork(OBS_DIM, N_ACTIONS, self.agent_hidden).to(self.device)
        net.load_state_dict(self._pool[idx])
        net.eval()
        self._current_net = net

    def _select_idx(self) -> int:
        n = len(self._pool)

        all_hist = []
        for h in self._win_hist:
            if len(h) >= self.MIN_MATCHES:
                all_hist.extend(h)
        global_wr = float(np.mean(all_hist)) if all_hist else 0.5

        if global_wr < self.WEAK_THRESH:
            return 0
        if global_wr > self.STRONG_THRESH:
            return n - 1

        if global_wr >= 0.5:
            weights = np.array([i + 1 for i in range(n)], dtype=float)
        else:
            weights = np.array([n - i for i in range(n)], dtype=float)
        weights /= weights.sum()
        return int(np.random.choice(n, p=weights))

    # -----------------------------------------------------------------------

    def act(self, obs_dict: dict) -> dict:
        """
        Red ajanlar için Q-argmax → ActionMapper aksiyon üret.
        obs_dict: DogfightEnv.reset/step'ten gelen normalize obs (tüm ajanlar).
        """
        actions = {}
        with torch.no_grad():
            for rid in self.red_ids:
                obs = obs_dict.get(rid)
                if obs is None or not np.all(np.isfinite(obs)):
                    actions[rid] = np.zeros(ACTION_DIM, dtype=np.float32)
                    continue
                obs_t  = torch.from_numpy(
                    obs[:OBS_DIM].astype(np.float32)
                ).unsqueeze(0).to(self.device)
                q_vals = self._current_net(obs_t)          # (1, N_ACTIONS)
                idx    = int(q_vals.argmax(dim=-1).item())
                actions[rid] = self.action_mapper(idx)
        return actions

    def record_outcome(self, is_win: bool) -> None:
        if self._current_idx is not None and not self._use_fallback:
            self._win_hist[self._current_idx].append(float(is_win))

    @property
    def using_fallback(self) -> bool:
        return self._use_fallback

    def log_status(self) -> str:
        if not self._pool:
            return "[QMIXPool] Pool boş — heuristic fallback aktif"
        lines = [f"[QMIXPool] {len(self._pool)} snapshot:"]
        all_hist = []
        for i, h in enumerate(self._win_hist):
            wr  = float(np.mean(h)) if h else 0.0
            n   = len(h)
            bar = "#" * int(wr * 20)
            lines.append(f"  snap-{i:2d}  wr={wr:.2f} (n={n:>2}) |{bar:<20}|")
            all_hist.extend(h)
        gwr = float(np.mean(all_hist)) if all_hist else float("nan")
        lines.append(f"  {'GENEL':20} wr={gwr:.2f}")
        return "\n".join(lines)


# ===========================================================================
# Replay Buffer
# ===========================================================================

class ReplayBuffer:
    """Individual-transition replay buffer."""

    def __init__(self, capacity: int = 50_000):
        self.capacity = capacity
        self.ptr      = 0
        self.size     = 0

        self.obs            = np.zeros((capacity, N_AGENTS, OBS_DIM),   dtype=np.float32)
        self.global_obs     = np.zeros((capacity, GLOBAL_OBS_DIM),      dtype=np.float32)
        self.actions        = np.zeros((capacity, N_AGENTS),            dtype=np.int64)
        self.rewards        = np.zeros((capacity, N_AGENTS),            dtype=np.float32)
        self.next_obs       = np.zeros((capacity, N_AGENTS, OBS_DIM),   dtype=np.float32)
        self.next_global_obs= np.zeros((capacity, GLOBAL_OBS_DIM),      dtype=np.float32)
        self.dones          = np.zeros((capacity,),                     dtype=np.float32)

    def add(self, obs, global_obs, actions, rewards,
            next_obs, next_global_obs, done) -> None:
        self.obs[self.ptr]             = obs
        self.global_obs[self.ptr]      = global_obs
        self.actions[self.ptr]         = actions
        self.rewards[self.ptr]         = rewards
        self.next_obs[self.ptr]        = next_obs
        self.next_global_obs[self.ptr] = next_global_obs
        self.dones[self.ptr]           = float(done)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        idxs = np.random.randint(0, self.size, batch_size)
        return {
            "obs":             self.obs[idxs],
            "global_obs":      self.global_obs[idxs],
            "actions":         self.actions[idxs],
            "rewards":         self.rewards[idxs],
            "next_obs":        self.next_obs[idxs],
            "next_global_obs": self.next_global_obs[idxs],
            "dones":           self.dones[idxs],
        }

    def __len__(self) -> int:
        return self.size


# ===========================================================================
# QMIX Trainer
# ===========================================================================

class QMIXTrainer:

    def __init__(self, config: dict, args: argparse.Namespace):
        self.config = config
        self.args   = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Hiperparametreler ────────────────────────────────────────────
        qcfg = config.get("qmix", {})
        self.lr                    = float(qcfg.get("lr",                    1e-4))
        self.gamma                 = float(qcfg.get("gamma",                 0.99))
        self.batch_size            = int(  qcfg.get("batch_size",            64))
        self.min_buffer            = int(  qcfg.get("min_buffer",            1000))
        self.eps_start             = float(qcfg.get("eps_start",             1.0))
        self.eps_end               = float(qcfg.get("eps_end",               0.05))
        self.eps_decay_steps       = int(  qcfg.get("eps_decay_steps",       100_000))
        self.target_update_interval= int(  qcfg.get("target_update_interval",200))
        self.train_freq            = int(  qcfg.get("train_freq",            4))
        self.grad_clip             = float(qcfg.get("grad_clip",             10.0))
        self.qmix_hidden           = int(  qcfg.get("qmix_hidden",          64))
        self.agent_hidden          = int(  qcfg.get("agent_hidden",         128))
        self.replay_capacity       = int(  qcfg.get("replay_capacity",      50_000))
        self.total_episodes        = int(  qcfg.get("total_episodes",       30_000))
        self.log_interval          = int(  qcfg.get("log_interval",         100))
        self.save_interval         = int(  qcfg.get("save_interval",        500))
        self.pool_update_interval  = int(  qcfg.get("pool_update_interval", 200))

        if args.test:
            self.total_episodes = 10
            self.min_buffer     = 0
            self.log_interval   = 1

        # ── Ortam — Faz-1 ile başlat ────────────────────────────────────
        self.env = DogfightEnv(config)
        start_phase = getattr(args, "start_phase", 1)
        self.env.set_curriculum_phase(start_phase)
        self.blue_ids = self.env.blue_ids
        self.red_ids  = self.env.red_ids
        print(f"[QMIX] Device      : {self.device}")
        print(f"[QMIX] obs_dim     : {OBS_DIM}")
        print(f"[QMIX] n_actions   : {N_ACTIONS}")
        print(f"[QMIX] n_agents    : {N_AGENTS}")

        # ── Heuristic Fallback ───────────────────────────────────────────
        all_ids  = self.blue_ids + self.red_ids
        team_map = {aid: ("blue" if aid in self.blue_ids else "red") for aid in all_ids}
        self.heuristic = MultiHeuristicPolicy(config, all_ids, team_map)

        # ── Aksiyon Haritası ─────────────────────────────────────────────
        self.action_mapper = ActionMapper()
        assert self.action_mapper.n_actions == N_ACTIONS

        # ── Ağlar ────────────────────────────────────────────────────────
        self.agent_net    = AgentQNetwork(OBS_DIM, N_ACTIONS, self.agent_hidden).to(self.device)
        self.target_net   = AgentQNetwork(OBS_DIM, N_ACTIONS, self.agent_hidden).to(self.device)
        self.mixer        = QMixNet(N_AGENTS, GLOBAL_OBS_DIM, self.qmix_hidden).to(self.device)
        self.target_mixer = QMixNet(N_AGENTS, GLOBAL_OBS_DIM, self.qmix_hidden).to(self.device)
        self._sync_targets()

        total_params = (
            sum(p.numel() for p in self.agent_net.parameters()) +
            sum(p.numel() for p in self.mixer.parameters())
        )
        print(f"[QMIX] Toplam parametre: {total_params:,}")

        # ── Optimizer ────────────────────────────────────────────────────
        self.optimizer = torch.optim.Adam(
            list(self.agent_net.parameters()) + list(self.mixer.parameters()),
            lr=self.lr,
        )

        # ── Buffer ───────────────────────────────────────────────────────
        self.buffer = ReplayBuffer(self.replay_capacity)

        # ── Curriculum ───────────────────────────────────────────────────
        self.curriculum = CurriculumManager(config)
        self.curriculum.phase = start_phase
        if start_phase == 2:
            self.env.set_dynamic_spawn_dist(self.curriculum.current_spawn_dist)

        # ── Self-Play Pool ───────────────────────────────────────────────
        self.pool = QMIXPool(
            red_ids      = self.red_ids,
            agent_hidden = self.agent_hidden,
            device       = self.device,
            fallback     = self.heuristic,
            action_mapper= self.action_mapper,
        )

        # ── Log / Checkpoint ─────────────────────────────────────────────
        self.ckpt_dir  = ROOT / "checkpoints"
        self.ckpt_dir.mkdir(exist_ok=True)
        self.log_path  = ROOT / "logs" / "qmix_log.csv"
        self.log_path.parent.mkdir(exist_ok=True)
        self._init_csv()

        # ── Sayaçlar ─────────────────────────────────────────────────────
        self.episode_count = 0
        self.total_steps   = 0
        self.losses: list  = []

        # Resume
        if args.resume:
            self._load_checkpoint(args.resume)
            # --start-phase geçersiz kılma: resume önce yükler, sonra fazı set eder
            if hasattr(args, "start_phase") and args.start_phase != 1:
                self.curriculum.phase = args.start_phase
                self.curriculum._ep_in_phase = 0
                self.curriculum._kill_history.clear()
                self.curriculum._win_history.clear()
                self.env.set_curriculum_phase(args.start_phase)
                if args.start_phase == 2:
                    self.env.set_dynamic_spawn_dist(self.curriculum.current_spawn_dist)
                print(f"[QMIX] Faz override: Faz-{args.start_phase} "
                      f"({self.curriculum.PHASE_NAMES.get(args.start_phase)})")

    # -----------------------------------------------------------------------
    # Yardımcılar
    # -----------------------------------------------------------------------

    def _sync_targets(self) -> None:
        self.target_net.load_state_dict(self.agent_net.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _epsilon(self) -> float:
        t = min(1.0, self.total_steps / max(self.eps_decay_steps, 1))
        return self.eps_start + (self.eps_end - self.eps_start) * t

    def _build_obs_array(self, obs_dict: dict) -> np.ndarray:
        arr = np.zeros((N_AGENTS, OBS_DIM), dtype=np.float32)
        for i, aid in enumerate(self.blue_ids):
            if aid in obs_dict:
                arr[i] = obs_dict[aid][:OBS_DIM]
        return arr

    def _build_global_obs(self, obs_array: np.ndarray) -> np.ndarray:
        return obs_array.reshape(-1)

    def _select_actions(self, obs_array: np.ndarray, epsilon: float) -> np.ndarray:
        actions = np.zeros(N_AGENTS, dtype=np.int64)
        with torch.no_grad():
            obs_t  = torch.from_numpy(obs_array).to(self.device)
            q_vals = self.agent_net(obs_t).cpu().numpy()
        for i in range(N_AGENTS):
            if np.random.random() < epsilon:
                actions[i] = np.random.randint(N_ACTIONS)
            else:
                actions[i] = int(np.argmax(q_vals[i]))
        return actions

    def _red_actions(self, obs_dict: dict, states: dict) -> dict:
        """Pool veya heuristic ile red aksiyonlarını üret."""
        if self.pool.using_fallback:
            all_acts = self.heuristic.act(states)
            return {rid: all_acts[rid] for rid in self.red_ids if rid in all_acts}
        else:
            return self.pool.act(obs_dict)

    # -----------------------------------------------------------------------
    # Güncelleme
    # -----------------------------------------------------------------------

    def _update(self) -> float | None:
        if len(self.buffer) < max(self.batch_size, self.min_buffer):
            return None

        batch = self.buffer.sample(self.batch_size)
        bs    = self.batch_size

        obs_t   = torch.from_numpy(batch["obs"]).to(self.device)
        glob_t  = torch.from_numpy(batch["global_obs"]).to(self.device)
        act_t   = torch.from_numpy(batch["actions"]).to(self.device)
        rew_t   = torch.from_numpy(batch["rewards"]).to(self.device)
        nobs_t  = torch.from_numpy(batch["next_obs"]).to(self.device)
        nglob_t = torch.from_numpy(batch["next_global_obs"]).to(self.device)
        dones_t = torch.from_numpy(batch["dones"]).to(self.device)

        obs_flat  = obs_t.view(bs * N_AGENTS, OBS_DIM)
        q_all     = self.agent_net(obs_flat).view(bs, N_AGENTS, N_ACTIONS)
        q_taken   = q_all.gather(2, act_t.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            nobs_flat  = nobs_t.view(bs * N_AGENTS, OBS_DIM)
            q_next_all = self.target_net(nobs_flat).view(bs, N_AGENTS, N_ACTIONS)
            q_next_max = q_next_all.max(dim=2).values

        q_tot = self.mixer(q_taken, glob_t)
        with torch.no_grad():
            q_tot_tgt = self.target_mixer(q_next_max, nglob_t)

        r_team = rew_t.mean(dim=1)
        y      = (r_team + self.gamma * q_tot_tgt * (1.0 - dones_t)).detach()

        loss = F.mse_loss(q_tot, y)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.agent_net.parameters()) + list(self.mixer.parameters()),
            self.grad_clip,
        )
        self.optimizer.step()
        return float(loss.item())

    # -----------------------------------------------------------------------
    # Tek Episode Rollout
    # -----------------------------------------------------------------------

    def _run_episode(self) -> dict:
        obs_dict = self.env.reset()
        self.pool.reset()               # episode başı: opponent seç

        ep_rewards = {aid: 0.0 for aid in self.blue_ids}
        ep_kills   = 0
        ep_steps   = 0
        done       = False
        obs_arr    = self._build_obs_array(obs_dict)

        while not done:
            states   = self.env.get_all_states()
            epsilon  = self._epsilon()

            disc_actions = self._select_actions(obs_arr, epsilon)
            cont_actions = {
                aid: self.action_mapper(disc_actions[i])
                for i, aid in enumerate(self.blue_ids)
            }

            red_acts    = self._red_actions(obs_dict, states)
            action_dict = {**cont_actions, **red_acts}

            next_obs_dict, rew_dict, done_dict, info_dict = self.env.step(action_dict)
            done = bool(done_dict.get("__all__", False))

            next_obs_arr  = self._build_obs_array(next_obs_dict)
            glob_obs      = self._build_global_obs(obs_arr)
            next_glob_obs = self._build_global_obs(next_obs_arr)
            rewards_arr   = np.array(
                [rew_dict.get(aid, 0.0) for aid in self.blue_ids], dtype=np.float32
            )

            self.buffer.add(
                obs_arr, glob_obs, disc_actions, rewards_arr,
                next_obs_arr, next_glob_obs, done,
            )

            if self.total_steps % self.train_freq == 0:
                loss = self._update()
                if loss is not None:
                    self.losses.append(loss)

            for aid in self.blue_ids:
                ep_rewards[aid] += rew_dict.get(aid, 0.0)
            for aid in self.blue_ids:
                ep_kills += float(info_dict.get(aid, {}).get("r_kill", 0.0) > 0.0)

            ep_steps     += 1
            self.total_steps += 1
            obs_arr   = next_obs_arr
            obs_dict  = next_obs_dict

        is_win = (done_dict.get("winner") == "blue")
        self.pool.record_outcome(is_win)     # episode sonu: pool güncelle

        return {
            "reward":  float(np.mean(list(ep_rewards.values()))),
            "kills":   ep_kills,
            "is_win":  is_win,
            "steps":   ep_steps,
            "epsilon": self._epsilon(),
        }

    # -----------------------------------------------------------------------
    # Ana Döngü
    # -----------------------------------------------------------------------

    def train(self) -> None:
        print(f"\n[QMIX] Eğitim başlıyor — toplam {self.total_episodes} episode")
        print(f"[QMIX] Başlangıç fazı: {self.curriculum.PHASE_NAMES.get(self.curriculum.phase)}")
        print(f"[QMIX] Pool güncelleme aralığı: her {self.pool_update_interval} ep\n")

        win_buf  = []
        kill_buf = []
        rew_buf  = []
        len_buf  = []
        t0 = time.time()

        while self.episode_count < self.total_episodes:
            ep_info = self._run_episode()
            self.episode_count += 1

            kills  = ep_info["kills"]
            is_win = ep_info["is_win"]

            win_buf.append(float(is_win))
            kill_buf.append(kills)
            rew_buf.append(ep_info["reward"])
            len_buf.append(ep_info["steps"])

            # Curriculum güncelle
            self.curriculum.record_episode(kills, int(is_win))

            # Faz-1.5: env'e dinamik spawn mesafesini bildir
            if self.curriculum.phase == 2:
                self.env.set_dynamic_spawn_dist(self.curriculum.current_spawn_dist)

            # Pool snapshot
            if self.episode_count % self.pool_update_interval == 0:
                self.pool.add_checkpoint(self.agent_net)
                print(f"  [QMIXPool] Snapshot eklendi "
                      f"(ep={self.episode_count}, pool_size={self.pool.size})")

            # Curriculum geçiş kontrolü
            if self.curriculum.check_transition():
                old_phase = self.curriculum.phase
                self.curriculum.advance()
                new_phase = self.curriculum.phase
                self.env.set_curriculum_phase(new_phase)
                if new_phase == 2:
                    self.env.set_dynamic_spawn_dist(self.curriculum.current_spawn_dist)
                print(f"\n{'='*60}")
                print(f"[Curriculum] FAZ GEÇİŞİ: {old_phase} → {new_phase}")
                print(f"[Curriculum] {self.curriculum.PHASE_NAMES.get(new_phase)}")
                print(f"{'='*60}\n")
                if new_phase == 4:
                    print("=" * 60)
                    print("  *** FAZ-4'E ULAŞILDI! 2v2 EĞITIM BAŞLIYOR ***")
                    print("=" * 60)
                    self._save_checkpoint(tag=f"ep{self.episode_count}_faz4_entry")
                # Pool'u sıfırla (yeni faz — eski pool geçersiz)
                self.pool = QMIXPool(
                    red_ids      = self.red_ids,
                    agent_hidden = self.agent_hidden,
                    device       = self.device,
                    fallback     = self.heuristic,
                    action_mapper= self.action_mapper,
                )

            # Periyodik log
            if self.episode_count % self.log_interval == 0:
                win_rate  = float(np.mean(win_buf[-self.log_interval:]))
                kill_mean = float(np.mean(kill_buf[-self.log_interval:]))
                rew_mean  = float(np.mean(rew_buf[-self.log_interval:]))
                len_mean  = float(np.mean(len_buf[-self.log_interval:]))
                loss_mean = float(np.mean(self.losses[-500:])) if self.losses else float("nan")
                eps       = ep_info["epsilon"]
                elapsed   = time.time() - t0
                sps       = int(self.total_steps / max(elapsed, 1))

                print(
                    f"[Ep {self.episode_count:6d}|Faz-{self.curriculum.phase}] "
                    f"step={self.total_steps:,} | "
                    f"rew={rew_mean:7.2f} | "
                    f"W={win_rate:.2f} | "
                    f"kills={kill_mean:.2f} | "
                    f"len={len_mean:.0f} | "
                    f"loss={loss_mean:.4f} | "
                    f"eps={eps:.3f} | "
                    f"pool={self.pool.size} | "
                    f"{sps}sps"
                )
                print(f"  {self.curriculum.status_str()}")
                self._write_csv(win_rate, kill_mean, rew_mean, loss_mean, eps)

                if self.episode_count % (self.log_interval * 5) == 0:
                    print(self.pool.log_status())

            # Target sync
            if self.episode_count % self.target_update_interval == 0:
                self._sync_targets()

            # Checkpoint
            if self.episode_count % self.save_interval == 0:
                self._save_checkpoint()

        self._save_checkpoint(final=True)
        print("\n[QMIX] Eğitim tamamlandı.")

    # -----------------------------------------------------------------------
    # Checkpoint & Log
    # -----------------------------------------------------------------------

    def _save_checkpoint(self, final: bool = False, tag: str = "") -> None:
        if final:
            name = "qmix_final.pt"
        elif tag:
            name = f"qmix_{tag}.pt"
        else:
            name = f"qmix_ep{self.episode_count}.pt"
        path = self.ckpt_dir / name
        torch.save({
            "episode":          self.episode_count,
            "total_steps":      self.total_steps,
            "curriculum_phase": self.curriculum.phase,
            "agent_net":        self.agent_net.state_dict(),
            "mixer":            self.mixer.state_dict(),
            "target_net":       self.target_net.state_dict(),
            "target_mixer":     self.target_mixer.state_dict(),
            "optimizer":        self.optimizer.state_dict(),
        }, path)
        print(f"  [QMIX] Checkpoint kaydedildi: {path}")

    def _load_checkpoint(self, path: str) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.agent_net.load_state_dict(ck["agent_net"])
        self.mixer.load_state_dict(ck["mixer"])
        self.target_net.load_state_dict(ck["target_net"])
        self.target_mixer.load_state_dict(ck["target_mixer"])
        self.optimizer.load_state_dict(ck["optimizer"])
        self.episode_count = ck.get("episode", 0)
        self.total_steps   = ck.get("total_steps", 0)
        if "curriculum_phase" in ck:
            self.curriculum.phase = ck["curriculum_phase"]
        print(f"[QMIX] Checkpoint yüklendi: {path} "
              f"(ep={self.episode_count}, faz={self.curriculum.phase})")

    def _init_csv(self) -> None:
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["episode", "phase", "win_rate", "kill_per_ep",
                     "mean_reward", "loss", "epsilon"]
                )

    def _write_csv(self, win_rate, kill_mean, rew_mean, loss_mean, eps) -> None:
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                self.episode_count,
                self.curriculum.phase,
                f"{win_rate:.4f}",
                f"{kill_mean:.4f}",
                f"{rew_mean:.4f}",
                f"{loss_mean:.6f}",
                f"{eps:.4f}",
            ])


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="QMIX curriculum eğitim")
    parser.add_argument("--config",      default="configs/config.yaml")
    parser.add_argument("--resume",      default=None, type=str)
    parser.add_argument("--start-phase", default=1, type=int,
                        help="Başlangıç curriculum fazı (1-4)")
    parser.add_argument("--test",        action="store_true", help="10 episode test modu")
    args = parser.parse_args()

    cfg_path = ROOT / args.config
    with open(cfg_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    trainer = QMIXTrainer(config, args)
    trainer.train()


if __name__ == "__main__":
    main()
