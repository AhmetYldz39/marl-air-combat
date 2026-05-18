"""
gat_comm.py
===========
Graf Dikkat Ağı (Graph Attention Network) iletişim modülü.

Faz-2 takım koordinasyonu için: her mavi ajan, takım arkadaşından
16 boyutlu mesaj alır. Mesaj, grafın kenar özelliklerini
(mesafe, bearing, tehdit) ve komşu düğümün ego durumunu kullanır.

Mimari:
    Düğümler  : her ajan → ego obs (node_dim=17, cooldown dahil)
    Kenarlar  : her çift için (edge_dim=3)
                  [distance_norm, bearing_norm, threat_score]
    Başlıklar : 4 attention head × 4 boyut = 16D çıkış

İşlem:
    1. Query  = W_q(node_i)                       (H × D)
    2. Key    = W_k(concat(node_j, edge_ij))      (H × D)
    3. Value  = W_v(concat(node_j, edge_ij))      (H × D)
    4. attn   = softmax(Q·Kᵀ / √D) × mask        (ölü ajanlar maskelenir)
    5. msg_i  = concat(attn × V, tüm başlıklar)  → (msg_dim=16)
    6. out    = LayerNorm(W_out(msg_i))

Parametre sayısı: ~2K (hafif modül)

Bağımlılıklar:
    torch >= 2.0

Bu dosya değişirse etkilenen dosyalar:
    training/train_mappo.py  (Phase2Trainer, _extend_obs_phase2)
    configs/config.yaml      (communication bölümü)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GATComm(nn.Module):
    """
    2 ajanlı takım için Graf Dikkat iletişim modülü.

    n_agents=2 sabit (2v2 mavi takım). Ölü ajan otomatik maskelenir.

    Parameters
    ----------
    node_dim : int
        Düğüm özellik boyutu — ego obs (17D, cooldown dahil)
    edge_dim : int
        Kenar özellik boyutu — [distance_norm, bearing_norm, threat]
    n_heads  : int
        Dikkat başlığı sayısı
    msg_dim  : int
        Çıkış mesaj boyutu — n_heads × head_dim olmalı
    """

    def __init__(self,
                 node_dim: int = 17,
                 edge_dim: int = 3,
                 n_heads:  int = 4,
                 msg_dim:  int = 16):
        super().__init__()
        assert msg_dim % n_heads == 0, "msg_dim, n_heads'e tam bölünmeli"

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.n_heads  = n_heads
        self.msg_dim  = msg_dim
        self.head_dim = msg_dim // n_heads   # 4

        kv_in = node_dim + edge_dim  # 20

        # Q, K, V projeksiyonları — tüm başlıklar için tek Linear
        self.W_q   = nn.Linear(node_dim, msg_dim, bias=False)  # 17 → 16
        self.W_k   = nn.Linear(kv_in,    msg_dim, bias=False)  # 20 → 16
        self.W_v   = nn.Linear(kv_in,    msg_dim, bias=False)  # 20 → 16
        self.W_out = nn.Linear(msg_dim,  msg_dim, bias=True)   # 16 → 16
        self.norm  = nn.LayerNorm(msg_dim)

        self._init_weights()

    def _init_weights(self):
        for m in [self.W_q, self.W_k, self.W_v]:
            nn.init.xavier_uniform_(m.weight)
        nn.init.orthogonal_(self.W_out.weight, gain=0.1)
        nn.init.zeros_(self.W_out.bias)

    def forward(self,
                node_feats: torch.Tensor,
                edge_feats: torch.Tensor,
                alive_mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        node_feats : (batch, n_agents, node_dim)
            Her ajanın ego obs'u.
        edge_feats : (batch, n_agents, n_agents, edge_dim)
            edge_feats[:, i, j, :] = ajan-i'den ajan-j'ye kenar özellikleri.
        alive_mask : (batch, n_agents)
            1.0 = hayatta, 0.0 = ölü.

        Returns
        -------
        messages : (batch, n_agents, msg_dim)
            Her ajan için 16 boyutlu çıkış mesajı.
        """
        B, N, _ = node_feats.shape
        H = self.n_heads
        D = self.head_dim

        # Query: (B, N, H, D)
        Q = self.W_q(node_feats).view(B, N, H, D)

        # Key & Value: node_j || edge_ij → (B, N_i, N_j, kv_in)
        node_j   = node_feats.unsqueeze(1).expand(B, N, N, -1)
        kv_input = torch.cat([node_j, edge_feats], dim=-1)  # (B, N, N, kv_in)

        K = self.W_k(kv_input).view(B, N, N, H, D)  # (B, N_i, N_j, H, D)
        V = self.W_v(kv_input).view(B, N, N, H, D)

        # Attention scores: Q_i · K_ij / sqrt(D)  → (B, N_i, N_j, H)
        Q_exp  = Q.unsqueeze(2)                      # (B, N_i, 1, H, D)
        scores = (Q_exp * K).sum(-1) / (D ** 0.5)   # (B, N_i, N_j, H)

        # Ölü ajan maskesi
        mask   = alive_mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, N_j, 1)
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # Self-loop kapatma
        eye = torch.eye(N, device=node_feats.device, dtype=torch.bool)
        eye = eye.unsqueeze(0).unsqueeze(-1)        # (1, N, N, 1)
        scores = scores.masked_fill(eye, float('-inf'))

        attn = F.softmax(scores, dim=2)             # (B, N_i, N_j, H)
        attn = torch.nan_to_num(attn, nan=0.0)      # tüm j ölüyse NaN → 0

        # Ağırlıklı toplam: (B, N_i, H, D)
        attn_exp = attn.unsqueeze(-1)               # (B, N_i, N_j, H, 1)
        agg = (attn_exp * V).sum(dim=2)             # (B, N_i, H, D)

        # Başlıkları birleştir → (B, N_i, msg_dim)
        agg = agg.reshape(B, N, self.msg_dim)

        # Çıkış projeksiyonu + LayerNorm
        return self.norm(self.W_out(agg))           # (B, N_i, msg_dim)

    @torch.no_grad()
    def compute_messages(self,
                         ego_obs_list:  list,
                         edge_feats_np: np.ndarray,
                         alive_list:    list,
                         device:        torch.device) -> list:
        """
        Rollout collection için numpy → mesaj → numpy dönüşümü.

        Parameters
        ----------
        ego_obs_list  : list[np.ndarray] — her ajan için (node_dim,)
        edge_feats_np : np.ndarray       — (n_agents, n_agents, edge_dim)
        alive_list    : list[float]      — [1.0, 1.0] veya [1.0, 0.0]
        device        : torch.device

        Returns
        -------
        list[np.ndarray] — her ajan için (msg_dim,)
        """
        N = len(ego_obs_list)

        node_t = torch.tensor(
            np.stack(ego_obs_list), dtype=torch.float32, device=device
        ).unsqueeze(0)                                         # (1, N, node_dim)

        edge_t = torch.tensor(
            edge_feats_np, dtype=torch.float32, device=device
        ).unsqueeze(0)                                         # (1, N, N, edge_dim)

        alive_t = torch.tensor(
            alive_list, dtype=torch.float32, device=device
        ).unsqueeze(0)                                         # (1, N)

        msgs_t = self.forward(node_t, edge_t, alive_t)        # (1, N, msg_dim)
        msgs   = msgs_t.squeeze(0).cpu().numpy()               # (N, msg_dim)
        return [msgs[i] for i in range(N)]
