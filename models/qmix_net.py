"""
qmix_net.py
===========
QMIX ağ bileşenleri.

Bileşenler:
  ActionMapper   : ayrık indeks (0–161) → sürekli aksiyon (5D)
  AgentQNetwork  : obs(50D) → Q(162)  paylaşımlı MLP
  QMixNet        : [Q_1, Q_2] + global_state → Q_total  (monotonicity garantili)

Aksiyon grid'i  (3×3×3×3×2 = 162):
  aileron  (DA): {-1,  0,  1}
  elevator (DE): {-1,  0,  1}
  rudder   (DR): {-1,  0,  1}
  throttle (DT): { 0, 0.5, 1}
  fire     (DF): { 0,  1}
"""

import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Aksiyon Haritası
# ---------------------------------------------------------------------------

class ActionMapper:
    """
    Ayrık indeksi (0–161) sürekli 5D kontrol vektörüne çevirir.
    Lookup tablosu itertools.product ile oluşturulur — sabit, parametre yok.
    """
    _DA = [-1.0,  0.0,  1.0]
    _DE = [-1.0,  0.0,  1.0]
    _DR = [-1.0,  0.0,  1.0]
    _DT = [ 0.0,  0.5,  1.0]
    _DF = [ 0.0,  1.0]

    def __init__(self):
        self.table = np.array(
            list(itertools.product(self._DA, self._DE, self._DR, self._DT, self._DF)),
            dtype=np.float32,
        )                              # (162, 5)
        self.n_actions = len(self.table)   # 162

    def __call__(self, idx: int) -> np.ndarray:
        return self.table[idx]

    def batch(self, idxs: np.ndarray) -> np.ndarray:
        return self.table[idxs]        # (N, 5)


# ---------------------------------------------------------------------------
# Ajan Q-Ağı
# ---------------------------------------------------------------------------

class AgentQNetwork(nn.Module):
    """
    Paylaşımlı ajan Q-ağı.
    obs(50D) → Linear(128) → ReLU → Linear(128) → ReLU → Q(162)
    """
    def __init__(self, obs_dim: int = 50, n_actions: int = 162, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (..., obs_dim)  →  (..., n_actions)
        return self.net(obs)


# ---------------------------------------------------------------------------
# Mixing Network
# ---------------------------------------------------------------------------

class QMixNet(nn.Module):
    """
    Monotonicity-garantili mixing network.

    Forward: [Q_1, Q_2] + global_state → Q_total
    Hiperağ ağırlıkları abs() ile pozitif tutulur → ∂Q_tot/∂Q_i ≥ 0.
    """
    def __init__(
        self,
        n_agents:       int = 2,
        global_obs_dim: int = 100,
        qmix_hidden:    int = 64,
    ):
        super().__init__()
        self.n_agents    = n_agents
        self.qmix_hidden = qmix_hidden

        # Katman-1 hiperağları
        self.hw1 = nn.Linear(global_obs_dim, n_agents * qmix_hidden)
        self.hb1 = nn.Linear(global_obs_dim, qmix_hidden)
        # Katman-2 hiperağları
        self.hw2 = nn.Linear(global_obs_dim, qmix_hidden)
        self.hb2 = nn.Linear(global_obs_dim, 1)

    def forward(
        self,
        agent_qs:     torch.Tensor,   # (batch, n_agents)
        global_state: torch.Tensor,   # (batch, global_obs_dim)
    ) -> torch.Tensor:                # (batch,)
        bs = agent_qs.size(0)
        x  = agent_qs.unsqueeze(1)                                      # (bs, 1, n_agents)

        # Katman 1
        w1 = torch.abs(self.hw1(global_state)).view(bs, self.n_agents, self.qmix_hidden)
        b1 = self.hb1(global_state).view(bs, 1, self.qmix_hidden)
        h  = F.elu(torch.bmm(x, w1) + b1)                              # (bs, 1, qmix_hidden)

        # Katman 2
        w2 = torch.abs(self.hw2(global_state)).view(bs, self.qmix_hidden, 1)
        b2 = self.hb2(global_state).view(bs, 1, 1)
        q_total = torch.bmm(h, w2) + b2                                 # (bs, 1, 1)
        return q_total.view(bs)                                          # (bs,)
