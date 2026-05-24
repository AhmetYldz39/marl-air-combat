"""
facmac_net.py
=============
FACMAC ağ bileşenleri.

  FACMACActor    : obs(50D) → ctrl(4D, clamp) — fire kural-tabanlı (ağ dışı)
  FACMACActorOM  : obs(60D) → ctrl(4D) — OM+Role split input; base(50)+intent(6)+role(4)
  FACMACTwinCritic : [obs(obs_dim) + action(action_dim)] → (Q1, Q2) scalar, TD3 twin critic
  QMixNet      : models/qmix_net.py'den import edilir (değişmeden kullanılır)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FACMACActor(nn.Module):
    """
    Decentralized FACMAC actor — yürütme sırasında sadece kendi obs'unu kullanır.

    Çıkış:
        ctrl (4D) : aileron, elevator, rudder → clamp(-1,1)
                    throttle                  → clamp(0,1)
    Fire kural-tabanlı: WEZ içi + cooldown==0 → fire=1, değilse fire=0
    """

    def __init__(self, obs_dim: int = 50, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden,  hidden), nn.ReLU(),
        )
        self.ctrl_head = nn.Linear(hidden, 4)   # aileron, elevator, rudder, throttle

    # -----------------------------------------------------------------------

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Döndürür: (..., 4) — clamp uygulanmış ctrl"""
        h    = self._features(obs)
        ctrl = self.ctrl_head(h)
        return torch.cat([
            ctrl[..., :3].clamp(-1.0, 1.0),   # aileron, elevator, rudder
            ctrl[..., 3:4].clamp(0.0,  1.0),  # throttle
        ], dim=-1)

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        """Rollout için: 4D ctrl döndürür. Fire harici kural tarafından belirlenir."""
        return self.forward(obs), None

    def action_for_grad(self, obs: torch.Tensor) -> torch.Tensor:
        """Actor update için differentiable 4D aksiyon."""
        return self.forward(obs)


class FACMACActorOM(nn.Module):
    """
    FACMAC actor — Centralized OM + Role split giriş.

    Obs yapısı (60D):
      [0:50]  fc1_base   — base obs (facmac_ep2000.pt net.0'dan transfer)
      [50:56] fc1_intent — OM intent 6D (sıfır init)
      [56:60] fc1_role   — rol ataması 4D (sıfır init)

    Transfer: fc1_base ← net.0, fc2 ← net.2, ctrl_head ← ctrl_head (FACMACActor'dan)
    """

    def __init__(self, base_obs_dim: int = 50,
                 intent_dim: int = 6, role_dim: int = 4,
                 hidden: int = 256):
        super().__init__()
        self.base_obs_dim = base_obs_dim
        self._s0 = base_obs_dim
        self._s1 = base_obs_dim + intent_dim

        self.fc1_base   = nn.Linear(base_obs_dim, hidden)
        self.fc1_intent = nn.Linear(intent_dim,   hidden, bias=False)
        self.fc1_role   = nn.Linear(role_dim,     hidden, bias=False)
        self.fc2        = nn.Linear(hidden, hidden)
        self.ctrl_head  = nn.Linear(hidden, 4)

        nn.init.zeros_(self.fc1_intent.weight)
        nn.init.zeros_(self.fc1_role.weight)

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        feat = F.relu(
            self.fc1_base(obs[..., :self._s0]) +
            self.fc1_intent(obs[..., self._s0:self._s1]) +
            self.fc1_role(obs[..., self._s1:])
        )
        return F.relu(self.fc2(feat))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h    = self._features(obs)
        ctrl = self.ctrl_head(h)
        return torch.cat([
            ctrl[..., :3].clamp(-1.0, 1.0),
            ctrl[..., 3:4].clamp(0.0,  1.0),
        ], dim=-1)

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        return self.forward(obs), None

    def action_for_grad(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs)


class FACMACTwinCritic(nn.Module):
    """
    TD3 twin critic — Q-value aşırı tahmin sorununu minimize eder.
    İki bağımsız Q-ağı; target Q = min(Q1_tot, Q2_tot) ile actor güncellenir.

    Giriş  : [obs(obs_dim) | action(action_dim)]
    Çıkış  : (Q1, Q2) — her biri (...,) scalar
    """

    def __init__(self, obs_dim: int = 50, action_dim: int = 4, hidden: int = 256):
        super().__init__()
        in_dim = obs_dim + action_dim

        def _build():
            return nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1),
            )

        self.q1 = _build()
        self.q2 = _build()

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        """
        obs    : (..., obs_dim)
        action : (..., action_dim)
        return : (Q1, Q2) — her biri (...,) scalar
        """
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def Q1_only(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Actor update için yalnızca Q1 — Q2 forward'u atlanır."""
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x).squeeze(-1)
