"""
om_net.py
=========
Centralized Opponent Modelling + Role Assignment.

  EnemyHistoryBuffer       — 20-adım sliding window (48D/adım)
  CentralizedOpponentModel — 960→256→128→6, denetimli intent tahmini
  CentralizedRoleAssigner  — sıralı Gumbel-Softmax, çakışma kısıtlı
  get_om_label()           — aspect-angle tabanlı intent supervision
  build_team_state()       — ham state'ten 6D takım vektörü
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Intent sınıfları ───────────────────────────────────────────────────────
INTENT_AGGRESSIVE = 0
INTENT_EVASIVE    = 1
INTENT_DEFENSIVE  = 2
N_INTENT_CLASSES  = 3

# ─── Rol sınıfları ──────────────────────────────────────────────────────────
ROLE_SNIPER    = 0
ROLE_PURSUIT   = 1
ROLE_DEFENSIVE = 2
ROLE_SUPPORT   = 3
N_ROLES        = 4

# Geçerli (r0, r1) rol çiftleri — P(4,2) = 12 kombinasyon
ROLE_PAIRS: list = [(i, j) for i in range(N_ROLES) for j in range(N_ROLES) if i != j]

# ─── Geçmiş tampon boyutları ────────────────────────────────────────────────
HISTORY_LEN = 20
N_ENEMIES   = 2
N_BLUE      = 2
ENEMY_SLOT  = 12   # normalization.py: OBS_ENEMY_DIM = 12
STEP_DIM    = N_ENEMIES * N_BLUE * ENEMY_SLOT   # 48
HISTORY_DIM = HISTORY_LEN * STEP_DIM            # 960

# OM obs uzantısı: intent(6D) + rol(4D) = 10D
OM_EXT_DIM    = N_ENEMIES * N_INTENT_CLASSES + N_ROLES  # 10
INTENT_DIM    = N_ENEMIES * N_INTENT_CLASSES              # 6
ROLE_DIM      = N_ROLES                                   # 4

# Düşman slot indeksleri (50D base obs: 17 ego + 9 tm + 12+12 enemies)
_EGO_DIM    = 17
_TM_DIM     = 9
_E0_START   = _EGO_DIM + _TM_DIM                          # 26
_E1_START   = _E0_START + ENEMY_SLOT                      # 38

# Intent threshold'ları
_ATA_AGG_RAD = np.radians(45.0)   # ≤45° → saldırgan
_ATA_EVA_RAD = np.radians(90.0)   # >90° → kaçınma

# Takım durumu normalleştirme
_AMMO_MAX = 10.0


# ===========================================================================
# EnemyHistoryBuffer
# ===========================================================================

class EnemyHistoryBuffer:
    """
    Centralized 20-adım düşman gözlem geçmişi.

    Her adımda:
      enemy_i_step = cat(blue_0_obs_of_enemy_i, blue_1_obs_of_enemy_i) = 24D
      step = cat(enemy_0_step, enemy_1_step) = 48D

    Episode başında sıfır-doldurma uygulanır.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._buf: deque = deque(
            [np.zeros(STEP_DIM, dtype=np.float32)] * HISTORY_LEN,
            maxlen=HISTORY_LEN,
        )

    def update(self, obs_arr: np.ndarray) -> np.ndarray:
        """
        obs_arr : (N_BLUE, base_obs_dim≥50) — blue ajanlarının temel obs'ları
        returns : (HISTORY_DIM,) = 960D düzleştirilmiş geçmiş
        """
        e0 = np.concatenate([
            obs_arr[0, _E0_START : _E0_START + ENEMY_SLOT],
            obs_arr[1, _E0_START : _E0_START + ENEMY_SLOT],
        ])  # 24D
        e1 = np.concatenate([
            obs_arr[0, _E1_START : _E1_START + ENEMY_SLOT],
            obs_arr[1, _E1_START : _E1_START + ENEMY_SLOT],
        ])  # 24D
        self._buf.append(np.concatenate([e0, e1]))      # 48D
        return np.concatenate(list(self._buf))           # 960D

    def get(self) -> np.ndarray:
        """Mevcut 960D geçmişi döndür (güncelleme yapmadan)."""
        return np.concatenate(list(self._buf))


# ===========================================================================
# CentralizedOpponentModel
# ===========================================================================

class CentralizedOpponentModel(nn.Module):
    """
    Düşman intent tahmini.
    960D geçmiş → 6D logit (2 düşman × 3 sınıf: agg/eva/def).
    Denetimli: CrossEntropy ile aspect-angle etiketlere karşı.

    Çıkış yapısı:
      logits[:, :3] → düşman 0 intent logit'leri
      logits[:,  3:] → düşman 1 intent logit'leri
    """

    def __init__(self, history_dim: int = HISTORY_DIM, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(history_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 128),         nn.ReLU(),
            nn.Linear(128, N_ENEMIES * N_INTENT_CLASSES),  # 6
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """history: (..., 960) → logits: (..., 6)"""
        return self.net(history)

    def intent_flat(self, history: torch.Tensor) -> torch.Tensor:
        """
        obs'a eklenmek üzere düzleştirilmiş softmax olasılıkları döndürür.
        returns: (..., 6) — [p_agg0, p_eva0, p_def0, p_agg1, p_eva1, p_def1]
        """
        logits = self.forward(history)
        p0 = F.softmax(logits[..., :3], dim=-1)
        p1 = F.softmax(logits[..., 3:], dim=-1)
        return torch.cat([p0, p1], dim=-1)  # (..., 6)

    def supervised_loss(self, history: torch.Tensor,
                        labels: torch.Tensor) -> torch.Tensor:
        """
        history : (batch, 960)
        labels  : (batch, 2) — düşman başına int label [0,1,2]
        returns : scalar CrossEntropy kaybı
        """
        logits = self.forward(history)  # (batch, 6)
        loss0  = F.cross_entropy(logits[:, :3], labels[:, 0])
        loss1  = F.cross_entropy(logits[:, 3:], labels[:, 1])
        return (loss0 + loss1) * 0.5


# ===========================================================================
# CentralizedRoleAssigner
# ===========================================================================

class CentralizedRoleAssigner(nn.Module):
    """
    Joint pair softmax ile çakışmasız 2-ajan rol ataması.

    12 geçerli çift (4P2 = 4×3) üzerinde tek Gumbel-Softmax:
      valid_pairs = [(i,j) for i in range(4) for j in range(4) if i != j]
      mlp(x) → 12 logit → Gumbel-Softmax → pair probs → marginal role_0, role_1

    Eski sequential yaklaşımda excl_mask pursuit'i sistematik olarak
    bastırıyordu; joint pair'de tüm çiftler eşit şansla seçilebilir.

    x = intent(6D) + team_state(6D) = 12D
    rol sınıfları: 0=sniper, 1=pursuit, 2=defensive, 3=support
    """

    # P(4,2) = 12 geçerli çift — sıra önemli (agent_0, agent_1)
    _PAIRS = [(i, j) for i in range(4) for j in range(4) if i != j]
    N_PAIRS = 12

    def __init__(self, input_dim: int = INTENT_DIM + 6, hidden: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, self.N_PAIRS),
        )
        # Çift indeks tablosu — (12, 2): [r0_idx, r1_idx]
        pairs_t = torch.tensor(self._PAIRS, dtype=torch.long)   # (12, 2)
        self.register_buffer("pair_indices", pairs_t)

    def _marginals(self, pair_probs: torch.Tensor):
        """pair_probs (..., 12) → role_0 (..., 4), role_1 (..., 4) marginal."""
        r0 = self.pair_indices[:, 0]   # (12,)
        r1 = self.pair_indices[:, 1]   # (12,)
        shape = pair_probs.shape[:-1]
        role_0 = torch.zeros(*shape, N_ROLES, device=pair_probs.device)
        role_1 = torch.zeros(*shape, N_ROLES, device=pair_probs.device)
        for r in range(N_ROLES):
            role_0[..., r] = pair_probs[..., r0 == r].sum(dim=-1)
            role_1[..., r] = pair_probs[..., r1 == r].sum(dim=-1)
        return role_0, role_1

    def forward(self, x: torch.Tensor,
                tau: float = 1.0, hard: bool = False):
        """
        x        : (..., 12)
        returns  : (role_0, role_1) — her biri (..., 4) marginal dağılım
        """
        logits     = self.mlp(x)                                    # (..., 12)
        pair_probs = F.gumbel_softmax(logits, tau=tau, hard=hard)  # (..., 12)
        return self._marginals(pair_probs)

    @torch.no_grad()
    def assign(self, x: torch.Tensor):
        """Eval zamanı deterministik rol ataması."""
        logits   = self.mlp(x)
        pair_idx = logits.argmax(dim=-1)                            # (...,)
        r0_idx   = self.pair_indices[pair_idx, 0]
        r1_idx   = self.pair_indices[pair_idx, 1]
        role_0   = F.one_hot(r0_idx, N_ROLES).float()
        role_1   = F.one_hot(r1_idx, N_ROLES).float()
        return role_0, role_1


# ===========================================================================
# Yardımcı Fonksiyonlar
# ===========================================================================

def get_om_label(enemy_state: np.ndarray, blue_states: list) -> int:
    """
    Tek bir düşman için intent label döndürür.

    enemy_state : ham aircraft state dizisi
    blue_states : [(state_array, alive_float), ...] — blue ajanları listesi
    returns     : int — INTENT_AGGRESSIVE / INTENT_EVASIVE / INTENT_DEFENSIVE
    """
    from envs.aircraft_model import STATE_X, STATE_Y, STATE_H, STATE_PSI, STATE_ALIVE
    from envs.geometry_utils import antenna_train_angle, distance_3d  # noqa: F401

    if float(enemy_state[STATE_ALIVE]) < 0.5:
        return INTENT_DEFENSIVE

    e_pos = enemy_state[[STATE_X, STATE_Y, STATE_H]]
    e_psi = float(enemy_state[STATE_PSI])

    min_ata = float("inf")
    for b_state, b_alive in blue_states:
        if float(b_alive) < 0.5:
            continue
        b_pos = b_state[[STATE_X, STATE_Y, STATE_H]]
        ata   = abs(antenna_train_angle(e_pos, b_pos, e_psi))
        if ata < min_ata:
            min_ata = ata

    if min_ata == float("inf"):   # tüm blue'lar ölü
        return INTENT_DEFENSIVE
    if min_ata <= _ATA_AGG_RAD:
        return INTENT_AGGRESSIVE
    if min_ata > _ATA_EVA_RAD:
        return INTENT_EVASIVE
    return INTENT_DEFENSIVE


def build_team_state(blue_states: list,
                     ammo_max: float = _AMMO_MAX) -> np.ndarray:
    """
    blue_states : [state_array_0, state_array_1] — None kabul edilir
    returns     : (6,) [hp0, ammo0_norm, alive0, hp1, ammo1_norm, alive1]
    """
    from envs.aircraft_model import STATE_HP, STATE_AMMO, STATE_ALIVE

    out = np.zeros(6, dtype=np.float32)
    for i, st in enumerate(blue_states):
        if st is None:
            continue
        base = i * 3
        out[base]     = float(st[STATE_HP])
        out[base + 1] = min(float(st[STATE_AMMO]) / ammo_max, 1.0)
        out[base + 2] = float(st[STATE_ALIVE])
    return out


# ===========================================================================
# OM Replay Buffer — supervised güncelleme için
# ===========================================================================

class OMReplayBuffer:
    """
    (history, label_enemy0, label_enemy1) geçişlerini saklar.
    CentralizedOpponentModel denetimli güncellemesinde kullanılır.
    """

    def __init__(self, capacity: int = 5_000) -> None:
        self.capacity = capacity
        self.ptr      = 0
        self.size     = 0
        self._hist   = np.zeros((capacity, HISTORY_DIM), dtype=np.float32)
        self._labels = np.zeros((capacity, N_ENEMIES), dtype=np.int64)

    def add(self, history: np.ndarray, label0: int, label1: int) -> None:
        self._hist[self.ptr]   = history
        self._labels[self.ptr] = [label0, label1]
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple:
        """returns: (history_np, labels_np)"""
        idxs = np.random.randint(0, self.size, batch_size)
        return self._hist[idxs], self._labels[idxs]

    def __len__(self) -> int:
        return self.size
