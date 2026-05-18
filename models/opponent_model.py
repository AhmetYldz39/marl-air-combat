"""
opponent_model.py
=================
Düşman intent modeli (OpponentModel) ve taktiksel rol seçici (RoleSelector).

OpponentModel:
    Son 20 adım düşman obs geçmişinden 2 düşman için 3-sınıflı intent tahmini.
    Girdi : (batch, 480)  ← HISTORY_STEPS × N_ENEMIES × ENEMY_OBS_DIM = 20 × 2 × 12
    Çıktı : (batch, 6)    ← 2 düşman × 3 sınıf, her blok ayrı softmax
    Sınıflar: 0=agresif  1=defansif  2=kaçma

    Auxiliary loss: OpponentModel.aux_loss(logits, rule_labels)
    Kural etiketleri: OpponentModel._make_label(history_np)

RoleSelector:
    Düşman intent + takım bağlamı + kaynaklar → 4 taktiksel rol.
    Girdi : (batch, 13) = opp_intent(6) + teammate_role(4) + resources(3)
    Çıktı : (batch, 4) Gumbel-Softmax
    Roller: 0=sniper  1=pursuit  2=defensive  3=support
    Eğitim: hard=False (soft, differentiable)
    Inference: hard=True (one-hot, discrete)

Entegrasyon (train_mappo.py):
    - OpponentModel trainer tarafından çağrılır, çıktı obs'a (68D→74D) eklenir
    - RoleSelector GATMAPPOActor içinde yaşar, actor.compute_role() ile çağrılır
    - Auxiliary loss opt_opp optimizer ile PPO'dan bağımsız güncellenir
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# normalization.py enemy_obs() ile aynı indeksler
_IDX_REL_V   = 3    # göreceli hız [-1, 1]  (enemy_V - agent_V)
_IDX_BEARING = 4    # bearing [-1, 1]
_IDX_ATA     = 6    # Antenna Train Angle [-1, 1]  (Blue'nun hedefe ATA'sı)
_IDX_AA      = 7    # Aspect angle [0, 1]  (düşmanın burnu Blue'ya açısı; 0=tam önde, 1=tam arkada)
_IDX_DIST    = 8    # mesafe [0, 1]  (wez_range_max = 8000m ile normalize)
_IDX_THREAT  = 9    # Blue'nun Red'e saldırı avantajı [0, 1]
_IDX_ALIVE   = 11   # {0, 1}

ENEMY_OBS_DIM = 12
N_ENEMIES     = 2
HISTORY_STEPS = 20
INTENT_DIM    = 3   # [agresif, defansif, kaçma]

# Sınıf indeksleri
AGGRESSIVE = 0
DEFENSIVE  = 1
EVADING    = 2


class OpponentModel(nn.Module):
    """2 düşman için 3-sınıflı intent tahmini (MLP üzerinde)."""

    def __init__(self,
                 history_steps: int = HISTORY_STEPS,
                 n_enemies:     int = N_ENEMIES,
                 enemy_obs_dim: int = ENEMY_OBS_DIM,
                 hidden1:       int = 128,
                 hidden2:       int = 64):
        super().__init__()
        self.history_steps = history_steps
        self.n_enemies     = n_enemies
        self.enemy_obs_dim = enemy_obs_dim
        self.in_dim        = history_steps * n_enemies * enemy_obs_dim  # 480

        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, n_enemies * INTENT_DIM),  # 6
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, history_flat: "torch.Tensor") -> "torch.Tensor":
        """
        history_flat: (batch, 480)
        Döndürür: (batch, 6) — her 3D blok ayrı softmax uygulanmış
        """
        logits = self.net(history_flat)
        parts  = [
            F.softmax(logits[..., i * INTENT_DIM:(i + 1) * INTENT_DIM], dim=-1)
            for i in range(self.n_enemies)
        ]
        return torch.cat(parts, dim=-1)

    def forward_with_logits(
            self, history_flat: "torch.Tensor"
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """(probs, logits) ikisi birden döndürür — aux_loss hesabı için."""
        logits = self.net(history_flat)
        parts  = [
            F.softmax(logits[..., i * INTENT_DIM:(i + 1) * INTENT_DIM], dim=-1)
            for i in range(self.n_enemies)
        ]
        return torch.cat(parts, dim=-1), logits

    @staticmethod
    def _make_label(history_np: np.ndarray) -> np.ndarray:
        """
        Kural tabanlı intent etiketi üretici (aux loss supervision).

        history_np : (HISTORY_STEPS, N_ENEMIES * ENEMY_OBS_DIM) = (20, 24)
        Döndürür   : (N_ENEMIES,) dtype=int64
                     0=agresif  1=defansif  2=kaçma

        heuristic_agent.py mod mantığına eşlenmiş kurallar (son 5 adım):

          Birincil sinyal: aspect angle (idx 7) — düşmanın burnu Blue'ya ne kadar dönük?
            0.0 = düşmanın tam önü Blue'ya bakıyor (PURSUIT modu)
            0.5 = dik açı (nötr)
            1.0 = düşmanın tam arkası Blue'ya dönük (kaçış modu)

          agresif : mean_aa < 0.33  →  Red burnu <%60 açıyla Blue'ya dönük → PURSUIT
          kaçma   : mean_aa > 0.67  →  Red burnu >120° Blue'dan uzak → EVASION/CRITICAL
          defansif: diğer           →  nötr geometri, orbital/bekleyiş
        """
        n_look = min(5, len(history_np))
        recent = history_np[-n_look:]                       # (≤5, 24)
        labels = np.ones(N_ENEMIES, dtype=np.int64)         # default: defansif

        for i in range(N_ENEMIES):
            s  = i * ENEMY_OBS_DIM
            sl = recent[:, s:s + ENEMY_OBS_DIM]            # (≤5, 12)

            mask = sl[:, _IDX_ALIVE] > 0.5
            if not mask.any():
                labels[i] = DEFENSIVE
                continue

            alive = sl[mask]

            # Aspect angle: düşmanın burnunun Blue'ya ne kadar dönük olduğu [0,1]
            # 0 = tam yüz yüze (pursuit), 1 = tam arkası dönük (kaçış)
            mean_aa = float(np.mean(alive[:, _IDX_AA]))

            # AGGRESSIVE: Red burnu Blue'ya dönük → PURSUIT moduna eşdeğer
            if mean_aa < 0.33:
                labels[i] = AGGRESSIVE
            # EVADING: Red burnu Blue'dan uzak → EVASION/CRITICAL moduna eşdeğer
            elif mean_aa > 0.67:
                labels[i] = EVADING
            else:
                labels[i] = DEFENSIVE

        return labels

    @staticmethod
    def aux_loss(logits: "torch.Tensor",
                 labels: "torch.Tensor") -> "torch.Tensor":
        """
        Auxiliary CrossEntropy loss.

        logits : (batch, N_ENEMIES * INTENT_DIM) = (batch, 6)  — pre-softmax logits
        labels : (batch, N_ENEMIES)              = (batch, 2)  — int64 sınıf indeksi
        Döndürür: scalar loss
        """
        losses = [
            F.cross_entropy(
                logits[..., i * INTENT_DIM:(i + 1) * INTENT_DIM],
                labels[:, i]
            )
            for i in range(N_ENEMIES)
        ]
        return sum(losses) / N_ENEMIES


class RoleSelector(nn.Module):
    """
    Taktiksel rol seçici.

    Girdi : concat([opp_intent(6), teammate_role(4), resources(3)]) = 13D
    Çıktı : 4D Gumbel-Softmax
    Roller: 0=sniper  1=pursuit  2=defensive  3=support
    """

    ROLES   = ["sniper", "pursuit", "defensive", "support"]
    N_ROLES = 4

    def __init__(self,
                 intent_dim:   int = N_ENEMIES * INTENT_DIM,  # 6
                 teammate_dim: int = 4,
                 resource_dim: int = 3,
                 n_roles:      int = 4,
                 hidden:       int = 32):
        super().__init__()
        in_dim = intent_dim + teammate_dim + resource_dim   # 13
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_roles),
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self,
                opp_intent:    "torch.Tensor",
                teammate_role: "torch.Tensor",
                resources:     "torch.Tensor",
                hard:          bool = False) -> "torch.Tensor":
        """
        opp_intent    : (batch, 6)
        teammate_role : (batch, 4) — önceki adım takım arkadaşı rolü; bilinmiyorsa uniform
        resources     : (batch, 3) — [fuel_norm, ammo_norm, hp] (ego obs[13:16])
        hard          : False=eğitim (soft), True=inference (one-hot)
        Döndürür      : (batch, 4)
        """
        x = torch.cat([opp_intent, teammate_role, resources], dim=-1)
        return F.gumbel_softmax(self.net(x), tau=0.3, hard=hard, dim=-1)
