"""
train_facmac.py
===============
FACMAC-TD3 eğitim döngüsü — 4-fazlı curriculum + 2v2 hava muharebe ortamı.

Mimari:
  Actor          : FACMACActor, paylaşımlı
                   obs(50D) → ctrl(4D, clamp) + fire(Bernoulli head)
  Twin Critic    : FACMACTwinCritic — (Q1, Q2) scalar, paylaşımlı
                   [obs(50D) + action(5D)] → (Q1, Q2)
  Mixer          : QMixNet × 2 (mixer1, mixer2) — monotonicity garantisi
                   target_Q = min(Q1_tot_tgt, Q2_tot_tgt)

TD3 iyileştirmeleri:
  1. Twin critics    : min(Q1, Q2) ile target → Q-overestimation baskılanır
  2. Delayed actor   : Actor her policy_freq=2 critic adımında güncellenir
  3. Target smoothing: next_action += clamp(N(0,σ=0.2), -0.5, 0.5) ctrl'e eklenir
  4. Grad clipping   : critic ve actor için ayrı clip_grad_norm_

Curriculum (4 faz):
  Faz-1: 1v1 WEZ-yakın  → kill/ep≥0.30 AND win≥0.40
  Faz-2: 1v1 kademeli   → kill/ep≥0.20, min 1200 ep
  Faz-3: 1v1 normal     → kill/ep≥0.15, min 300 ep
  Faz-4: 2v2 normal     (son faz) + FACMACPool self-play

Pool (Faz-4):
  FACMACPool — actor snapshot ring buffer (max 20)
  Adaptif seçim: global_wr < 0.2 → eski, > 0.6 → yeni, diğer → ağırlıklı random

Kullanım:
    python -u -X utf8 training/train_facmac.py
    python -u -X utf8 training/train_facmac.py --test
    python -u -X utf8 training/train_facmac.py --start-phase 4
    python -u -X utf8 training/train_facmac.py --resume checkpoints/facmac_ep5000.pt --start-phase 4
"""

import argparse
import copy
import csv
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.dogfight_env import DogfightEnv
from agents.heuristic_agent import MultiHeuristicPolicy
from models.facmac_net import FACMACActor, FACMACActorOM, FACMACTwinCritic
from models.qmix_net import QMixNet
from models.om_net import (
    CentralizedOpponentModel, CentralizedRoleAssigner,
    EnemyHistoryBuffer, OMReplayBuffer,
    build_team_state, get_om_label, INTENT_DEFENSIVE, ROLE_PAIRS,
)
from envs.aircraft_model import (STATE_X, STATE_Y, STATE_H,
                                   STATE_PSI, STATE_AMMO, STATE_ALIVE)
from envs.geometry_utils import antenna_train_angle, distance_3d


# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

N_AGENTS       = 2
BASE_OBS_DIM   = 50                          # env obs boyutu
OBS_DIM        = 60                          # actor/critic obs boyutu: 50+6+4
ACTION_DIM     = 4   # fire kural-tabanlı; actor sadece ctrl (aileron,elevator,rudder,throttle)
GLOBAL_OBS_DIM = N_AGENTS * OBS_DIM   # 120

_WEZ_RANGE_MIN = 300.0
_WEZ_RANGE_MAX = 8000.0
_WEZ_ANGLE_MAX = np.radians(30.0)


# ===========================================================================
# Curriculum Manager
# ===========================================================================

class CurriculumManager:
    PHASE_NAMES = {
        1: "Faz-1   (1v1 WEZ-close)",
        2: "Faz-1.5 (1v1 dinamik-dist)",
        3: "Faz-2   (1v1 normal)",
        4: "Faz-3   (2v2 normal)",
    }

    def __init__(self, config: dict):
        cur = config.get("curriculum_v2", {})
        self.phase = 1
        self.eval_window             = int(cur.get("eval_window",              100))
        self.phase1_kill_thresh      = float(cur.get("phase1_kill_threshold",  0.30))
        self.phase1_win_thresh       = float(cur.get("phase1_win_threshold",   0.40))
        self.phase1_min_ep           = int(cur.get("phase1_min_episodes",      100))
        self.phase15_dist_start      = float(cur.get("phase15_dist_start",    4000.0))
        self.phase15_dist_max        = float(cur.get("phase15_dist_max",     16000.0))
        self.phase15_dist_step       = float(cur.get("phase15_dist_step",     1000.0))
        self.phase15_step_eps        = int(cur.get("phase15_step_episodes",    100))
        self.phase15_pullback_thresh = float(cur.get("phase15_pullback_thresh", 0.10))
        self.phase15_kill_thresh     = float(cur.get("phase15_kill_threshold",  0.20))
        self.phase15_min_ep          = int(cur.get("phase15_min_episodes",     1200))
        self.phase2_kill_thresh      = float(cur.get("phase2_kill_threshold",   0.15))
        self.phase2_min_ep           = int(cur.get("phase2_min_episodes",       300))
        self._kill_history: list     = []
        self._win_history:  list     = []
        self._ep_in_phase:  int      = 0
        self.current_spawn_dist      = self.phase15_dist_start

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
                    self.current_spawn_dist - self.phase15_dist_step)
                print(f"[Curriculum] Faz-1.5: kill/ep={recent_kill:.3f} → "
                      f"mesafe geri: {self.current_spawn_dist:.0f}m")
            else:
                self.current_spawn_dist = min(
                    self.phase15_dist_max,
                    self.current_spawn_dist + self.phase15_dist_step)
                print(f"[Curriculum] Faz-1.5: kill/ep={recent_kill:.3f} → "
                      f"mesafe: {self.current_spawn_dist:.0f}m")

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
        rk = float(np.mean(self._kill_history[-w:])) if n > 0 else 0.0
        rw = float(np.mean(self._win_history[-w:]))  if n > 0 else 0.0
        if self.phase == 1:
            ts = f"kill>={self.phase1_kill_thresh:.2f} win>={self.phase1_win_thresh:.2f}"
        elif self.phase == 2:
            ts = f"kill>={self.phase15_kill_thresh:.2f} dist={self.current_spawn_dist:.0f}m"
        elif self.phase == 3:
            ts = f"kill>={self.phase2_kill_thresh:.2f} min={self.phase2_min_ep}"
        else:
            ts = "son faz"
        return (f"{self.PHASE_NAMES.get(self.phase)} | ep_in_phase={self._ep_in_phase} | "
                f"kill/ep={rk:.3f} win={rw:.3f} [{ts}]")


# ===========================================================================
# FACMACPool — Faz-4 Adaptif Self-Play
# ===========================================================================

class FACMACPool:
    WIN_WINDOW    = 20
    WEAK_THRESH   = 0.2
    STRONG_THRESH = 0.6
    MIN_MATCHES   = 5

    def __init__(self, red_ids: list, obs_dim: int, hidden: int,
                 device: torch.device, max_size: int = 20):
        self.red_ids      = red_ids
        self.obs_dim      = BASE_OBS_DIM  # pool her zaman 50D base actor kullanır
        self.hidden       = hidden
        self.device       = device
        self.max_size     = max_size
        self._snapshots:   list = []
        self._win_records: list = []
        self._current_idx  = -1
        self._current_actor = None
        self.using_fallback = True

    def add_checkpoint(self, actor) -> None:
        sd = copy.deepcopy(actor.state_dict())
        # FACMACActorOM → FACMACActor key remap (pool her zaman 50D base actor kullanır)
        if 'fc1_base.weight' in sd:
            remap = {
                'fc1_base.weight': 'net.0.weight',
                'fc1_base.bias':   'net.0.bias',
                'fc2.weight':      'net.2.weight',
                'fc2.bias':        'net.2.bias',
                'ctrl_head.weight': 'ctrl_head.weight',
                'ctrl_head.bias':   'ctrl_head.bias',
            }
            sd = {remap[k]: v for k, v in sd.items() if k in remap}
        if len(self._snapshots) >= self.max_size:
            self._snapshots.pop(0)
            self._win_records.pop(0)
        self._snapshots.append(sd)
        self._win_records.append(deque(maxlen=self.WIN_WINDOW))

    def _global_wr(self) -> float:
        eligible = [r for r in self._win_records if len(r) >= self.MIN_MATCHES]
        if not eligible:
            return 0.5
        return float(np.mean([np.mean(list(r)) for r in eligible]))

    def reset(self) -> None:
        if not self._snapshots:
            self.using_fallback = True
            self._current_idx   = -1
            self._current_actor = None
            return
        self.using_fallback = False
        n   = len(self._snapshots)
        gwr = self._global_wr()
        if gwr < self.WEAK_THRESH:
            idx = 0
        elif gwr > self.STRONG_THRESH:
            idx = n - 1
        else:
            weights = np.ones(n, dtype=np.float64) / n
            idx = int(np.random.choice(n, p=weights))
        self._current_idx = idx
        net = FACMACActor(self.obs_dim, self.hidden).to(self.device)
        net.load_state_dict(self._snapshots[idx])
        net.eval()
        self._current_actor = net

    def act(self, obs_dict: dict) -> dict:
        if self.using_fallback or self._current_actor is None:
            return {}
        actions = {}
        with torch.no_grad():
            for rid in self.red_ids:
                if rid not in obs_dict:
                    continue
                obs_np = np.array(obs_dict[rid], dtype=np.float32)[:self.obs_dim]
                obs_t  = torch.from_numpy(obs_np).unsqueeze(0).to(self.device)
                act, _ = self._current_actor.act(obs_t, deterministic=False)
                ctrl4 = act.squeeze(0).cpu().numpy()           # 4D ctrl
                actions[rid] = np.append(ctrl4, 0.0)          # 5D: fire=0 (pool ateş etmez)
        return actions

    def record_outcome(self, is_win: bool) -> None:
        if self._current_idx >= 0 and not self.using_fallback:
            self._win_records[self._current_idx].append(float(is_win))

    def log_status(self) -> str:
        n   = len(self._snapshots)
        gwr = self._global_wr() if n > 0 else 0.0
        return (f"Pool: {n} ckpt, gwr={gwr:.3f}, "
                f"fallback={self.using_fallback}, idx={self._current_idx}")


# ---------------------------------------------------------------------------
# Replay Buffer  (continuous actions)
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity: int = 10_000,
                 obs_dim: int = OBS_DIM, global_obs_dim: int = GLOBAL_OBS_DIM):
        self.capacity    = capacity
        self.obs_dim     = obs_dim
        self.glob_dim    = global_obs_dim
        self.ptr  = 0
        self.size = 0
        self.obs             = np.zeros((capacity, N_AGENTS, obs_dim),    dtype=np.float32)
        self.global_obs      = np.zeros((capacity, global_obs_dim),       dtype=np.float32)
        self.actions         = np.zeros((capacity, N_AGENTS, ACTION_DIM), dtype=np.float32)
        self.rewards         = np.zeros((capacity, N_AGENTS),             dtype=np.float32)
        self.next_obs        = np.zeros((capacity, N_AGENTS, obs_dim),    dtype=np.float32)
        self.next_global_obs = np.zeros((capacity, global_obs_dim),       dtype=np.float32)
        self.dones           = np.zeros((capacity,),                      dtype=np.float32)

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


# ---------------------------------------------------------------------------
# FACMAC-TD3 Trainer
# ---------------------------------------------------------------------------

class FACMACTrainer:

    def __init__(self, config: dict, args: argparse.Namespace):
        self.config = config
        self.args   = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Hiperparametreler ────────────────────────────────────────────
        fcfg = config.get("facmac", {})
        self.lr_actor            = float(fcfg.get("lr_actor",            1e-4))
        self.lr_critic           = float(fcfg.get("lr_critic",           1e-3))
        self.gamma               = float(fcfg.get("gamma",               0.99))
        self.tau                 = float(fcfg.get("tau",                 0.005))
        self.batch_size          = int(  fcfg.get("batch_size",          256))
        self.min_buffer          = int(  fcfg.get("min_buffer",          1000))
        self.noise_sigma         = float(fcfg.get("noise_sigma",         0.1))
        self.noise_clip          = float(fcfg.get("noise_clip",          0.5))
        self.noise_decay         = float(fcfg.get("noise_decay",         0.9999))
        self.grad_clip           = float(fcfg.get("grad_clip",           10.0))
        self.hidden              = int(  fcfg.get("hidden",              256))
        self.qmix_hidden         = int(  fcfg.get("qmix_hidden",         64))
        self.replay_capacity     = int(  fcfg.get("replay_capacity",     10_000))
        self.train_freq          = int(  fcfg.get("train_freq",          4))
        self.total_episodes      = int(  fcfg.get("total_episodes",      30_000))
        self.log_interval        = int(  fcfg.get("log_interval",        100))
        self.save_interval       = int(  fcfg.get("save_interval",       500))
        self.policy_freq         = int(  fcfg.get("policy_freq",         2))
        self.target_noise_sigma  = float(fcfg.get("target_noise_sigma",  0.2))
        self.target_noise_clip   = float(fcfg.get("target_noise_clip",   0.5))
        self.pool_update_interval = (999_999 if getattr(args, "no_pool", False)
                                     else int(fcfg.get("pool_update_interval", 500)))
        self.q_target_clamp      = float(fcfg.get("q_target_clamp",      2000.0))

        if getattr(args, "total_episodes", None) is not None:
            self.total_episodes = args.total_episodes
        if args.test:
            self.total_episodes = 10
            self.min_buffer     = 0
            self.log_interval   = 1

        # ── Ortam ────────────────────────────────────────────────────────
        self.env = DogfightEnv(config)
        start_phase = getattr(args, "start_phase", 1)
        self.env.set_curriculum_phase(start_phase)
        self.blue_ids = self.env.blue_ids
        self.red_ids  = self.env.red_ids

        all_ids  = self.blue_ids + self.red_ids
        team_map = {aid: ("blue" if aid in self.blue_ids else "red") for aid in all_ids}
        self.heuristic = MultiHeuristicPolicy(config, all_ids, team_map)

        # ── Curriculum ───────────────────────────────────────────────────
        self.curriculum = CurriculumManager(config)
        self.curriculum.phase = start_phase
        if start_phase == 2:
            self.env.set_dynamic_spawn_dist(self.curriculum.current_spawn_dist)

        # ── Centralized OM + Role bileşenleri ───────────────────────────
        self.om_mode = getattr(args, "mode", "facmac_om") == "facmac_om"
        if self.om_mode:
            self.cent_om          = CentralizedOpponentModel().to(self.device)
            self.cent_role        = CentralizedRoleAssigner().to(self.device)
            self.enemy_hist       = EnemyHistoryBuffer()
            self.om_replay        = OMReplayBuffer(capacity=5_000)
            self.opt_om           = torch.optim.Adam(self.cent_om.parameters(), lr=1e-3)
            self.opt_role         = torch.optim.Adam(self.cent_role.parameters(), lr=3e-4)
            self._role_inp_buf:   list  = []
            self._role_pair_buf:  list  = []
            self._return_history        = deque(maxlen=200)
        else:
            self.cent_om = self.cent_role = self.enemy_hist = self.om_replay = None

        # ── Ağlar (TD3: twin critics + iki mixer) ────────────────────────
        if self.om_mode:
            self.actor_obs_dim  = OBS_DIM         # 60D
            self.actor = FACMACActorOM(BASE_OBS_DIM, hidden=self.hidden).to(self.device)
        else:
            self.actor_obs_dim  = BASE_OBS_DIM    # 50D — pure FACMAC
            self.actor = FACMACActor(BASE_OBS_DIM, hidden=self.hidden).to(self.device)
        self.global_obs_dim = N_AGENTS * self.actor_obs_dim
        self.actor_target  = copy.deepcopy(self.actor)
        self.critic        = FACMACTwinCritic(self.actor_obs_dim, ACTION_DIM, self.hidden).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.mixer1        = QMixNet(N_AGENTS, self.global_obs_dim, self.qmix_hidden).to(self.device)
        self.mixer1_target = copy.deepcopy(self.mixer1)
        self.mixer2        = QMixNet(N_AGENTS, self.global_obs_dim, self.qmix_hidden).to(self.device)
        self.mixer2_target = copy.deepcopy(self.mixer2)

        for net in (self.actor_target, self.critic_target,
                    self.mixer1_target, self.mixer2_target):
            for p in net.parameters():
                p.requires_grad_(False)

        n_act = sum(p.numel() for p in self.actor.parameters())
        n_cri = sum(p.numel() for p in self.critic.parameters())
        n_mx  = sum(p.numel() for p in self.mixer1.parameters()) * 2
        print(f"[FACMAC-TD3] Device      : {self.device}")
        print(f"[FACMAC-TD3] Params      : actor={n_act:,}  twin_critic={n_cri:,}  "
              f"2×mixer={n_mx:,}  total={n_act+n_cri+n_mx:,}")
        print(f"[FACMAC-TD3] Başlangıç   : {self.curriculum.PHASE_NAMES.get(start_phase)}")
        print(f"[FACMAC-TD3] policy_freq={self.policy_freq} "
              f"target_noise_sigma={self.target_noise_sigma} "
              f"buffer_capacity={self.replay_capacity}")

        # ── Optimizer ────────────────────────────────────────────────────
        self.opt_actor  = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.opt_critic = torch.optim.Adam(
            list(self.critic.parameters()) +
            list(self.mixer1.parameters()) +
            list(self.mixer2.parameters()),
            lr=self.lr_critic,
        )

        # ── Buffer & Sayaçlar ────────────────────────────────────────────
        self.buffer          = ReplayBuffer(self.replay_capacity,
                                            obs_dim=self.actor_obs_dim,
                                            global_obs_dim=self.global_obs_dim)
        self.episode_count   = 0
        self.total_steps     = 0
        self.noise_sigma_    = self.noise_sigma
        self._update_counter = 0
        self.pool            = None   # FACMACPool — Faz-4'te başlatılır
        self.losses_critic: list = []
        self.losses_actor:  list = []
        self._intent_acc: list = []   # her adımda intent_np biriktirilir
        self._role_acc:   list = []   # her adımda role_0_np biriktirilir

        # ── Log ──────────────────────────────────────────────────────────
        self.ckpt_dir    = ROOT / "checkpoints"
        self.ckpt_dir.mkdir(exist_ok=True)
        self.ckpt_prefix = getattr(args, "checkpoint_prefix", "facmac_om")
        self.log_path = ROOT / "logs" / "facmac_log.csv"
        self.log_path.parent.mkdir(exist_ok=True)
        self._init_csv()

        # start_phase=4 ile doğrudan Faz-4 başlatma (resume olsun ya da olmasın)
        if start_phase == 4:
            self._init_pool()

        if args.resume:
            self._load_checkpoint(args.resume)
            if hasattr(args, "start_phase") and args.start_phase != 1:
                self.curriculum.phase = args.start_phase
                self.curriculum._ep_in_phase = 0
                self.curriculum._kill_history.clear()
                self.curriculum._win_history.clear()
                self.env.set_curriculum_phase(args.start_phase)
                self._refresh_agents()
                if args.start_phase == 2:
                    self.env.set_dynamic_spawn_dist(self.curriculum.current_spawn_dist)
                if args.start_phase == 4 and self.pool is None:
                    self._init_pool()

    # -----------------------------------------------------------------------
    # Yardımcılar
    # -----------------------------------------------------------------------

    def _init_pool(self) -> None:
        self.pool = FACMACPool(
            self.red_ids, BASE_OBS_DIM, self.hidden, self.device
        )
        print(f"[FACMAC-TD3] FACMACPool başlatıldı (red_ids={self.red_ids})")

    def _refresh_agents(self) -> None:
        """Faz geçişinde blue_ids/red_ids ve heuristic güncelle."""
        self.blue_ids = self.env.blue_ids
        self.red_ids  = self.env.red_ids
        all_ids  = self.blue_ids + self.red_ids
        team_map = {aid: ("blue" if aid in self.blue_ids else "red") for aid in all_ids}
        self.heuristic = MultiHeuristicPolicy(self.config, all_ids, team_map)

    def _polyak(self, online: nn.Module, target: nn.Module) -> None:
        for po, pt in zip(online.parameters(), target.parameters()):
            pt.data.copy_(self.tau * po.data + (1.0 - self.tau) * pt.data)

    def _build_obs_array(self, obs_dict: dict) -> np.ndarray:
        """50D base obs dizisi döndürür."""
        arr = np.zeros((N_AGENTS, BASE_OBS_DIM), dtype=np.float32)
        for i, aid in enumerate(self.blue_ids):
            if aid in obs_dict:
                arr[i] = np.array(obs_dict[aid], dtype=np.float32)[:BASE_OBS_DIM]
        return arr

    def _build_extended_obs(self, obs_arr: np.ndarray) -> np.ndarray:
        """
        om_mode=True : 50D base obs → 60D: base(50) + intent(6) + role(4).
        om_mode=False: obs_arr olduğu gibi döner (50D, OM yok).
        """
        if not self.om_mode:
            return obs_arr   # 50D olduğu gibi döner

        hist_960 = self.enemy_hist.update(obs_arr)

        with torch.no_grad():
            hist_t    = torch.from_numpy(hist_960).unsqueeze(0).to(self.device)
            intent_np = self.cent_om.intent_flat(hist_t).squeeze(0).cpu().numpy()  # (6,)

        # OM replay doldur
        if np.any(hist_960 != 0):
            states = self.env.get_all_states()
            blue_pairs = [
                (states.get(bid), float(states.get(bid)[STATE_ALIVE]))
                for bid in self.blue_ids
                if states.get(bid) is not None
            ]
            label0 = INTENT_DEFENSIVE
            label1 = INTENT_DEFENSIVE
            if len(self.red_ids) > 0:
                rs0 = states.get(self.red_ids[0])
                if rs0 is not None:
                    label0 = get_om_label(rs0, blue_pairs)
            if len(self.red_ids) > 1:
                rs1 = states.get(self.red_ids[1])
                if rs1 is not None:
                    label1 = get_om_label(rs1, blue_pairs)
            self.om_replay.add(hist_960, label0, label1)

        # Rol ataması
        states = self.env.get_all_states()
        ts_np  = build_team_state([states.get(bid) for bid in self.blue_ids])
        with torch.no_grad():
            x_role  = torch.from_numpy(
                np.concatenate([intent_np, ts_np]).astype(np.float32)
            ).unsqueeze(0).to(self.device)
            role_0, role_1 = self.cent_role.assign(x_role)
        role_0_np = role_0.squeeze(0).cpu().numpy()
        role_1_np = role_1.squeeze(0).cpu().numpy()
        roles = [role_0_np, role_1_np]

        # REINFORCE için çift indeksi ve girdi biriktir
        r0_idx   = int(role_0_np.argmax())
        r1_idx   = int(role_1_np.argmax())
        pair_idx = ROLE_PAIRS.index((r0_idx, r1_idx))
        self._role_pair_buf.append(pair_idx)
        x_role_np = np.concatenate([intent_np, ts_np]).astype(np.float32)
        self._role_inp_buf.append(x_role_np.copy())

        # Intent / rol birikimi (log için)
        self._intent_acc.append(intent_np.copy())
        self._role_acc.append(role_0_np.copy())

        # 60D extended obs
        ext = np.zeros((N_AGENTS, OBS_DIM), dtype=np.float32)
        for i in range(len(self.blue_ids)):
            ext[i] = np.concatenate([obs_arr[i], intent_np, roles[i]])
        return ext

    def _build_global_obs(self, obs_array: np.ndarray) -> np.ndarray:
        return obs_array.reshape(-1)

    def _select_actions(self, obs_array: np.ndarray) -> np.ndarray:
        """4D ctrl çıktısı + Gaussian keşif gürültüsü. Fire harici kural tarafından belirlenir."""
        n_blue  = len(self.blue_ids)
        actions = np.zeros((N_AGENTS, ACTION_DIM), dtype=np.float32)
        obs_t   = torch.from_numpy(obs_array).to(self.device)
        with torch.no_grad():
            for i in range(n_blue):
                act, _ = self.actor.act(obs_t[i:i+1], deterministic=False)
                act    = act.squeeze(0).cpu().numpy()
                noise  = np.clip(np.random.randn(4) * self.noise_sigma_,
                                 -self.noise_clip, self.noise_clip)
                act[0] = np.clip(act[0] + noise[0], -1.0, 1.0)
                act[1] = np.clip(act[1] + noise[1], -1.0, 1.0)
                act[2] = np.clip(act[2] + noise[2], -1.0, 1.0)
                act[3] = np.clip(act[3] + noise[3],  0.0, 1.0)
                actions[i] = act
        return actions

    def _rule_based_fire(self, obs_arr: np.ndarray, states: dict) -> np.ndarray:
        """WEZ içi + cooldown==0 ise fire=1, aksi halde fire=0."""
        fire = np.zeros(N_AGENTS, dtype=np.float32)
        for i, blue_id in enumerate(self.blue_ids):
            if obs_arr[i][16] > 1e-4:   # cooldown_norm > 0 → henüz hazır değil
                continue
            b = states.get(blue_id)
            if b is None or b[STATE_ALIVE] < 0.5 or b[STATE_AMMO] < 0.5:
                continue
            b_pos = b[[STATE_X, STATE_Y, STATE_H]]
            for red_id in self.red_ids:
                r = states.get(red_id)
                if r is None or r[STATE_ALIVE] < 0.5:
                    continue
                r_pos = r[[STATE_X, STATE_Y, STATE_H]]
                dist  = distance_3d(b_pos, r_pos)
                if dist < _WEZ_RANGE_MIN or dist > _WEZ_RANGE_MAX:
                    continue
                ata = antenna_train_angle(b_pos, r_pos, b[STATE_PSI])
                if abs(ata) <= _WEZ_ANGLE_MAX:
                    fire[i] = 1.0
                    break
        return fire

    def _red_actions(self, obs_dict: dict, states: dict) -> dict:
        """Faz-4'te pool; diğer fazlarda heuristic."""
        if (self.curriculum.phase == 4 and self.pool is not None
                and not self.pool.using_fallback):
            pool_acts = self.pool.act(obs_dict)
            if pool_acts:
                return pool_acts
        all_acts = self.heuristic.act(states)
        return {rid: all_acts[rid] for rid in self.red_ids if rid in all_acts}

    # -----------------------------------------------------------------------
    # REINFORCE Rol Gradyanı
    # -----------------------------------------------------------------------

    def _update_role_reinforce(self, episode_return: float) -> None:
        if not self.om_mode:
            return
        if len(self._role_pair_buf) < 1:
            self._role_pair_buf.clear(); self._role_inp_buf.clear()
            return
        self._return_history.append(episode_return)
        if len(self._return_history) < 10:
            self._role_pair_buf.clear(); self._role_inp_buf.clear()
            return
        ret_mean = float(np.mean(self._return_history))
        ret_std  = float(np.std(self._return_history)) + 1e-8
        ret_norm = float(np.clip((episode_return - ret_mean) / ret_std, -3.0, 3.0))
        x_np    = np.array(self._role_inp_buf, dtype=np.float32)
        x_t     = torch.FloatTensor(x_np).to(self.device)
        logits  = self.cent_role.mlp(x_t)
        log_p   = F.log_softmax(logits, dim=-1)
        pair_t  = torch.LongTensor(self._role_pair_buf).to(self.device)
        T       = len(pair_t)
        assigned_lp = log_p[torch.arange(T, device=self.device), pair_t]
        probs   = F.softmax(logits, dim=-1)
        entropy = -(probs * log_p).sum(dim=-1).mean()
        role_loss = -(ret_norm * assigned_lp.mean()) - 0.05 * entropy
        self.opt_role.zero_grad()
        role_loss.backward()
        nn.utils.clip_grad_norm_(self.cent_role.parameters(), 0.5)
        self.opt_role.step()
        self._role_pair_buf.clear()
        self._role_inp_buf.clear()

    # -----------------------------------------------------------------------
    # TD3 Güncelleme
    # -----------------------------------------------------------------------

    def _update(self) -> tuple:
        if len(self.buffer) < max(self.batch_size, self.min_buffer):
            return None, None

        batch = self.buffer.sample(self.batch_size)
        bs    = self.batch_size

        obs_t   = torch.from_numpy(batch["obs"]).to(self.device)              # (bs, N, OBS)
        glob_t  = torch.from_numpy(batch["global_obs"]).to(self.device)       # (bs, GLOB)
        acts_t  = torch.from_numpy(batch["actions"]).to(self.device)          # (bs, N, ACT)
        rew_t   = torch.from_numpy(batch["rewards"]).to(self.device)          # (bs, N)
        nobs_t  = torch.from_numpy(batch["next_obs"]).to(self.device)         # (bs, N, OBS)
        nglob_t = torch.from_numpy(batch["next_global_obs"]).to(self.device)  # (bs, GLOB)
        dones_t = torch.from_numpy(batch["dones"]).to(self.device)            # (bs,)

        # ── Critic güncelleme (her adım) ───────────────────────────────
        with torch.no_grad():
            nobs_flat = nobs_t.view(bs * N_AGENTS, self.actor_obs_dim)
            nact_flat = self.actor_target.action_for_grad(nobs_flat)          # (bs*N, 4)

            # Target policy smoothing — 4D ctrl (aileron/elevator/rudder/throttle)
            noise = torch.clamp(
                torch.randn(bs * N_AGENTS, 4, device=self.device) * self.target_noise_sigma,
                -self.target_noise_clip, self.target_noise_clip,
            )
            nact_smooth = nact_flat.clone()
            nact_smooth[:, :3] = torch.clamp(nact_flat[:, :3] + noise[:, :3], -1.0, 1.0)
            nact_smooth[:, 3]  = torch.clamp(nact_flat[:, 3]  + noise[:, 3],  0.0, 1.0)

            q1_tgt_flat, q2_tgt_flat = self.critic_target(nobs_flat, nact_smooth)
            q1_tgt    = q1_tgt_flat.view(bs, N_AGENTS)
            q2_tgt    = q2_tgt_flat.view(bs, N_AGENTS)
            q1_tot_tg = self.mixer1_target(q1_tgt, nglob_t)                   # (bs,)
            q2_tot_tg = self.mixer2_target(q2_tgt, nglob_t)                   # (bs,)
            q_tot_tgt = torch.min(q1_tot_tg, q2_tot_tg)
            r_team    = rew_t.mean(dim=1)
            y         = r_team + self.gamma * q_tot_tgt * (1.0 - dones_t)
            y         = y.clamp(-self.q_target_clamp, self.q_target_clamp)

        obs_flat  = obs_t.view(bs * N_AGENTS, self.actor_obs_dim)
        acts_flat = acts_t.view(bs * N_AGENTS, ACTION_DIM)
        q1_flat, q2_flat = self.critic(obs_flat, acts_flat)
        q1_online = q1_flat.view(bs, N_AGENTS)
        q2_online = q2_flat.view(bs, N_AGENTS)
        q1_tot    = self.mixer1(q1_online, glob_t)                            # (bs,)
        q2_tot    = self.mixer2(q2_online, glob_t)                            # (bs,)
        loss_critic = F.mse_loss(q1_tot, y) + F.mse_loss(q2_tot, y)

        self.opt_critic.zero_grad()
        loss_critic.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.critic.parameters()) +
            list(self.mixer1.parameters()) +
            list(self.mixer2.parameters()),
            self.grad_clip,
        )
        self.opt_critic.step()

        self._update_counter += 1
        loss_actor_val = float("nan")

        # ── Actor güncelleme (her policy_freq adımda) ──────────────────
        if self._update_counter % self.policy_freq == 0:
            cur_act_flat  = self.actor.action_for_grad(obs_flat)              # (bs*N, 4)
            q1_actor_flat = self.critic.Q1_only(obs_flat, cur_act_flat)       # (bs*N,)
            q1_actor      = q1_actor_flat.view(bs, N_AGENTS)
            q1_tot_actor  = self.mixer1(q1_actor, glob_t)                    # (bs,)
            loss_actor    = -q1_tot_actor.mean()

            self.opt_actor.zero_grad()
            loss_actor.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
            self.opt_actor.step()
            loss_actor_val = float(loss_actor.item())

            # Polyak — actor güncellendiğinde tüm target'lar güncellenir
            self._polyak(self.actor,  self.actor_target)
            self._polyak(self.critic, self.critic_target)
            self._polyak(self.mixer1, self.mixer1_target)
            self._polyak(self.mixer2, self.mixer2_target)

        # OM supervised güncelleme (PPO'dan bağımsız)
        if self.om_mode and len(self.om_replay) >= 32:
            batch_om = min(len(self.om_replay), 128)
            hist_np, labels_np = self.om_replay.sample(batch_om)
            hist_t  = torch.FloatTensor(hist_np).to(self.device)
            label_t = torch.LongTensor(labels_np).to(self.device)
            om_loss = self.cent_om.supervised_loss(hist_t, label_t)
            self.opt_om.zero_grad()
            om_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.cent_om.parameters(), 0.5)
            self.opt_om.step()

        return float(loss_critic.item()), loss_actor_val

    # -----------------------------------------------------------------------
    # Tek Episode Rollout
    # -----------------------------------------------------------------------

    def _run_episode(self) -> dict:
        obs_dict   = self.env.reset()
        self.heuristic.reset()
        if self.om_mode:
            self.enemy_hist.reset()
        if self.pool is not None:
            self.pool.reset()
        ep_rewards = {aid: 0.0 for aid in self.blue_ids}
        ep_kills   = 0
        ep_steps   = 0
        done       = False
        base_arr   = self._build_obs_array(obs_dict)       # (N, 50) — env obs
        obs_arr    = self._build_extended_obs(base_arr)    # (N, 60) — actor input

        while not done:
            states   = self.env.get_all_states()
            ctrl_arr = self._select_actions(obs_arr)            # (N, 4) — ctrl only
            fire_arr = self._rule_based_fire(base_arr, states)  # (N,)   — 0/1 (base obs)
            acts_5d  = np.concatenate(
                [ctrl_arr, fire_arr[:, np.newaxis]], axis=-1
            )                                                    # (N, 5) — env için
            cont_actions = {self.blue_ids[i]: acts_5d[i]
                            for i in range(len(self.blue_ids))}
            red_acts     = self._red_actions(obs_dict, states)
            action_dict  = {**cont_actions, **red_acts}

            next_obs_dict, rew_dict, done_dict, info_dict = self.env.step(action_dict)
            done = bool(done_dict.get("__all__", False))

            next_base_arr = self._build_obs_array(next_obs_dict)
            next_obs_arr  = self._build_extended_obs(next_base_arr)
            glob_obs      = self._build_global_obs(obs_arr)       # (120,)
            next_glob_obs = self._build_global_obs(next_obs_arr)  # (120,)
            rewards_arr   = np.zeros(N_AGENTS, dtype=np.float32)
            for i, aid in enumerate(self.blue_ids):
                rewards_arr[i] = rew_dict.get(aid, 0.0)

            # Buffer 60D extended obs + 4D ctrl saklar
            self.buffer.add(
                obs_arr, glob_obs, ctrl_arr, rewards_arr,
                next_obs_arr, next_glob_obs, done,
            )

            if self.total_steps % self.train_freq == 0:
                lc, la = self._update()
                if lc is not None:
                    self.losses_critic.append(lc)
                    if not np.isnan(la):
                        self.losses_actor.append(la)

            for aid in self.blue_ids:
                ep_rewards[aid] += rew_dict.get(aid, 0.0)
            for aid in self.blue_ids:
                ep_kills += float(info_dict.get(aid, {}).get("r_kill", 0.0) > 0.0)

            ep_steps     += 1
            self.total_steps += 1
            base_arr = next_base_arr
            obs_arr  = next_obs_arr
            obs_dict = next_obs_dict

        self.noise_sigma_ = max(0.01, self.noise_sigma_ * self.noise_decay)
        is_win = (done_dict.get("winner") == "blue")

        if self.pool is not None:
            self.pool.record_outcome(is_win)

        mean_rew = float(np.mean(list(ep_rewards.values())))
        if self.om_mode:
            self._update_role_reinforce(mean_rew)

        return {
            "reward":  mean_rew,
            "kills":   float(ep_kills),
            "is_win":  is_win,
            "steps":   ep_steps,
            "noise":   self.noise_sigma_,
        }

    # -----------------------------------------------------------------------
    # Ana Döngü
    # -----------------------------------------------------------------------

    def train(self) -> None:
        print(f"\n[FACMAC-TD3] Eğitim başlıyor — toplam {self.total_episodes} episode\n")

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

            self.curriculum.record_episode(kills, int(is_win))
            if self.curriculum.phase == 2:
                self.env.set_dynamic_spawn_dist(self.curriculum.current_spawn_dist)

            # Faz geçiş kontrolü
            if self.curriculum.check_transition():
                old_phase = self.curriculum.phase
                self.curriculum.advance()
                new_phase = self.curriculum.phase
                self.env.set_curriculum_phase(new_phase)
                self._refresh_agents()
                if new_phase == 2:
                    self.env.set_dynamic_spawn_dist(self.curriculum.current_spawn_dist)
                # Buffer sıfırla — stale dağılım karışmasını engelle
                self.buffer.ptr  = 0
                self.buffer.size = 0
                print(f"[FACMAC-TD3] Buffer sıfırlandı (Faz-{new_phase} için temiz başlangıç)")
                print(f"\n{'='*60}")
                print(f"[Curriculum] FAZ GEÇİŞİ: {old_phase} → {new_phase}")
                print(f"[Curriculum] {self.curriculum.PHASE_NAMES.get(new_phase)}")
                print(f"{'='*60}\n")
                if new_phase == 4:
                    print("=" * 60)
                    print("  *** FAZ-4'E ULAŞILDI! 2v2 EĞITIM BAŞLIYOR ***")
                    print("=" * 60)
                    self._init_pool()
                    self._save_checkpoint(tag=f"ep{self.episode_count}_faz4_entry")

            # Pool snapshot (sadece Faz-4)
            if (self.curriculum.phase == 4 and self.pool is not None
                    and self.curriculum._ep_in_phase > 0
                    and self.curriculum._ep_in_phase % self.pool_update_interval == 0):
                self.pool.add_checkpoint(self.actor)
                print(f"  [Pool] Snapshot eklendi — {self.pool.log_status()}")

            if self.episode_count % self.log_interval == 0:
                win_rate  = float(np.mean(win_buf[-self.log_interval:]))
                kill_mean = float(np.mean(kill_buf[-self.log_interval:]))
                rew_mean  = float(np.mean(rew_buf[-self.log_interval:]))
                len_mean  = float(np.mean(len_buf[-self.log_interval:]))
                lc_mean   = (float(np.mean(self.losses_critic[-500:]))
                             if self.losses_critic else float("nan"))
                la_mean   = (float(np.mean(self.losses_actor[-500:]))
                             if self.losses_actor  else float("nan"))
                elapsed   = time.time() - t0
                sps       = int(self.total_steps / max(elapsed, 1))

                pool_str = ""
                if self.pool is not None:
                    pool_str = f" | {self.pool.log_status()}"

                # Intent / rol istatistikleri
                if self._intent_acc:
                    intent_mat  = np.stack(self._intent_acc[-500:])   # (N, 3)
                    intent_mean = intent_mat.mean(axis=0)             # agg, def, eva
                    role_mat    = np.stack(self._role_acc[-500:])     # (N, 4)
                    role_idx    = role_mat.argmax(axis=1)
                    role_names  = ["sniper", "pursuit", "defensive", "support"]
                    role_dist   = {n: float((role_idx == i).mean())
                                   for i, n in enumerate(role_names)}
                else:
                    intent_mean = np.array([0.0, 0.0, 0.0])
                    role_dist   = {n: 0.0 for n in ["sniper", "pursuit", "defensive", "support"]}

                print(
                    f"[Ep {self.episode_count:6d}|Faz-{self.curriculum.phase}] "
                    f"step={self.total_steps:,} | "
                    f"rew={rew_mean:7.2f} | "
                    f"W={win_rate:.2f} | "
                    f"kills={kill_mean:.2f} | "
                    f"len={len_mean:.0f} | "
                    f"lc={lc_mean:.4f}  la={la_mean:.5f} | "
                    f"noise={ep_info['noise']:.3f} | "
                    f"{sps}sps{pool_str}"
                )
                print(
                    f"          intent: agg={intent_mean[0]:.2f} def={intent_mean[1]:.2f} "
                    f"eva={intent_mean[2]:.2f} | "
                    f"role: sniper={role_dist['sniper']:.2f} pursuit={role_dist['pursuit']:.2f} "
                    f"def={role_dist['defensive']:.2f} sup={role_dist['support']:.2f}"
                )
                print(f"  {self.curriculum.status_str()}")
                self._write_csv(win_rate, kill_mean, rew_mean, lc_mean, la_mean,
                                intent_mean, role_dist)

            if self.episode_count % self.save_interval == 0:
                self._save_checkpoint()

        self._save_checkpoint(final=True)
        print("\n[FACMAC-TD3] Eğitim tamamlandı.")

    # -----------------------------------------------------------------------
    # Checkpoint & Log
    # -----------------------------------------------------------------------

    def _save_checkpoint(self, final: bool = False, tag: str = "") -> None:
        pfx = self.ckpt_prefix
        if final:
            name = f"{pfx}_final.pt"
        elif tag:
            name = f"{pfx}_{tag}.pt"
        else:
            name = f"{pfx}_ep{self.episode_count}.pt"
        path = self.ckpt_dir / name
        torch.save({
            "episode":           self.episode_count,
            "total_steps":       self.total_steps,
            "curriculum_phase":  self.curriculum.phase,
            "_update_counter":   self._update_counter,
            "actor":             self.actor.state_dict(),
            "actor_target":      self.actor_target.state_dict(),
            "critic":            self.critic.state_dict(),
            "critic_target":     self.critic_target.state_dict(),
            "mixer1":            self.mixer1.state_dict(),
            "mixer1_target":     self.mixer1_target.state_dict(),
            "mixer2":            self.mixer2.state_dict(),
            "mixer2_target":     self.mixer2_target.state_dict(),
            "opt_actor":         self.opt_actor.state_dict(),
            "opt_critic":        self.opt_critic.state_dict(),
            "noise_sigma":       self.noise_sigma_,
        }, path)
        if self.om_mode:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            ck["cent_om"]   = self.cent_om.state_dict()
            ck["cent_role"] = self.cent_role.state_dict()
            ck["opt_role"]  = self.opt_role.state_dict()
            torch.save(ck, path)
        print(f"  [FACMAC-TD3] Checkpoint: {path}")

    def _load_actor_weights(self, actor_sd: dict) -> None:
        """FACMACActor(50D) veya FACMACActorOM(60D) checkpoint'ten yükler."""
        if 'fc1_base.weight' in actor_sd:
            # Yeni FACMACActorOM formatı — direkt yükle
            self.actor.load_state_dict(actor_sd, strict=False)
        else:
            # Eski FACMACActor(50D) formatı — key remap ile transfer
            remap = {
                'net.0.weight': 'fc1_base.weight',
                'net.0.bias':   'fc1_base.bias',
                'net.2.weight': 'fc2.weight',
                'net.2.bias':   'fc2.bias',
                'ctrl_head.weight': 'ctrl_head.weight',
                'ctrl_head.bias':   'ctrl_head.bias',
            }
            new_sd = {remap[k]: v for k, v in actor_sd.items() if k in remap}
            self.actor.load_state_dict(new_sd, strict=False)
            print("[FACMAC-TD3] Actor: FACMACActor(50D) → FACMACActorOM(60D) transfer; "
                  "fc1_intent/fc1_role sıfırdan başlıyor")

    def _load_checkpoint(self, path: str) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self._load_actor_weights(ck["actor"])
        actor_tgt_sd = ck.get("actor_target", ck["actor"])
        # actor_target için aynı remap
        if 'fc1_base.weight' in actor_tgt_sd:
            self.actor_target.load_state_dict(actor_tgt_sd, strict=False)
        else:
            remap = {'net.0.weight': 'fc1_base.weight', 'net.0.bias': 'fc1_base.bias',
                     'net.2.weight': 'fc2.weight', 'net.2.bias': 'fc2.bias',
                     'ctrl_head.weight': 'ctrl_head.weight', 'ctrl_head.bias': 'ctrl_head.bias'}
            self.actor_target.load_state_dict(
                {remap[k]: v for k, v in actor_tgt_sd.items() if k in remap}, strict=False
            )
        def _safe_load(module, sd):
            cur = module.state_dict()
            ok  = {k: v for k, v in sd.items()
                   if k in cur and v.shape == cur[k].shape}
            module.load_state_dict(ok, strict=False)

        _safe_load(self.critic,        ck["critic"])
        _safe_load(self.critic_target, ck["critic_target"])
        _safe_load(self.mixer1,        ck["mixer1"])
        _safe_load(self.mixer1_target, ck["mixer1_target"])
        _safe_load(self.mixer2,        ck["mixer2"])
        _safe_load(self.mixer2_target, ck["mixer2_target"])
        if self.om_mode:
            if "cent_om" in ck:
                self.cent_om.load_state_dict(ck["cent_om"])
            if "cent_role" in ck:
                try:
                    self.cent_role.load_state_dict(ck["cent_role"])
                except RuntimeError:
                    print("[FACMAC] cent_role mimari değişti — sıfırdan başlıyor (joint pair)")
            if "opt_role" in ck:
                try:
                    self.opt_role.load_state_dict(ck["opt_role"])
                except (ValueError, KeyError):
                    print("[FACMAC] opt_role state uyuşmadı — sıfırdan")
        try:
            self.opt_actor.load_state_dict(ck["opt_actor"])
            self.opt_critic.load_state_dict(ck["opt_critic"])
        except (ValueError, KeyError):
            print("[FACMAC-TD3] Optimizer state uyuşmadı — sıfırdan devam")
        self.episode_count   = ck.get("episode", 0)
        self.total_steps     = ck.get("total_steps", 0)
        self.noise_sigma_    = ck.get("noise_sigma", self.noise_sigma)
        self._update_counter = ck.get("_update_counter", 0)
        print(f"[FACMAC-TD3] Checkpoint yüklendi: {path} (ep={self.episode_count})")

    def _init_csv(self) -> None:
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["episode", "phase", "win_rate", "kill_per_ep", "mean_reward",
                     "loss_critic", "loss_actor",
                     "intent_agg", "intent_def", "intent_eva",
                     "role_sniper", "role_pursuit", "role_defensive", "role_support"]
                )

    def _write_csv(self, win_rate, kill_mean, rew_mean, lc, la,
                   intent_mean=None, role_dist=None) -> None:
        if intent_mean is None:
            intent_mean = np.zeros(3)
        if role_dist is None:
            role_dist = {"sniper": 0.0, "pursuit": 0.0, "defensive": 0.0, "support": 0.0}
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                self.episode_count, self.curriculum.phase,
                f"{win_rate:.4f}", f"{kill_mean:.4f}",
                f"{rew_mean:.4f}", f"{lc:.6f}", f"{la:.6f}",
                f"{intent_mean[0]:.4f}", f"{intent_mean[1]:.4f}", f"{intent_mean[2]:.4f}",
                f"{role_dist['sniper']:.4f}", f"{role_dist['pursuit']:.4f}",
                f"{role_dist['defensive']:.4f}", f"{role_dist['support']:.4f}",
            ])


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FACMAC-TD3 curriculum eğitim")
    parser.add_argument("--config",      default="configs/config.yaml")
    parser.add_argument("--resume",      default=None, type=str)
    parser.add_argument("--start-phase", default=1, type=int,
                        help="Başlangıç curriculum fazı (1-4)")
    parser.add_argument("--mode",              default="facmac_om",
                        choices=["facmac", "facmac_om"],
                        help="facmac: OM olmadan | facmac_om: OM+Rol (varsayılan)")
    parser.add_argument("--checkpoint-prefix", default=None, type=str,
                        help="Checkpoint dosya adı öneki (varsayılan: mod bazlı otomatik)")
    parser.add_argument("--seed",           default=None, type=int, help="Global random seed")
    parser.add_argument("--total-episodes", default=None, type=int, help="Eğitim episode sayısı override")
    parser.add_argument("--no-pool", action="store_true",
                        help="Pool devre dışı — yalnızca heuristic rakip")
    parser.add_argument("--test",    action="store_true", help="10 episode test modu")
    args = parser.parse_args()
    # Prefix otomatik belirleme — explicit override varsa kullan
    if args.checkpoint_prefix is None:
        args.checkpoint_prefix = "facmac_base" if args.mode == "facmac" else "facmac_om"
    # Seed
    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    cfg_path = ROOT / args.config
    with open(cfg_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    trainer = FACMACTrainer(config, args)
    trainer.train()


if __name__ == "__main__":
    main()
