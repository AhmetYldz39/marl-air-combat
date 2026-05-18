"""
facmac_net.py
=============
FACMAC ağ bileşenleri.

  FACMACActor  : obs(50D) → ctrl(4D, clamp) + fire(Bernoulli head)
  FACMACTwinCritic : [obs(50D) + action(5D)] → (Q1, Q2) scalar, TD3 twin critic
  QMixNet      : models/qmix_net.py'den import edilir (değişmeden kullanılır)
"""

import torch
import torch.nn as nn
from torch.distributions import Bernoulli


class FACMACActor(nn.Module):
    """
    Decentralized FACMAC actor — yürütme sırasında sadece kendi obs'unu kullanır.

    Çıkış:
        ctrl      (4D) : aileron, elevator, rudder → clamp(-1,1)
                         throttle                  → clamp(0,1)
        fire_head (1D) : ayrı Bernoulli logit head
                         bias = -0.85 → sigmoid(-0.85) ≈ 0.30 başlangıç ateş olasılığı
    """

    def __init__(self, obs_dim: int = 50, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden,  hidden), nn.ReLU(),
        )
        self.ctrl_head = nn.Linear(hidden, 4)   # aileron, elevator, rudder, throttle
        self.fire_head = nn.Linear(hidden, 1)   # fire logit
        nn.init.constant_(self.fire_head.bias, -0.85)

    # -----------------------------------------------------------------------

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def forward(self, obs: torch.Tensor):
        """
        Döndürür:
            ctrl_out   : (..., 4) — clamp uygulanmış, differentiable
            fire_prob  : (..., 1) — sigmoid(logit), differentiable
            fire_logit : (..., 1) — ham logit (Bernoulli dist için)
        """
        h    = self._features(obs)
        ctrl = self.ctrl_head(h)
        ctrl_out = torch.cat([
            ctrl[..., :3].clamp(-1.0, 1.0),   # aileron, elevator, rudder
            ctrl[..., 3:4].clamp(0.0,  1.0),  # throttle
        ], dim=-1)
        fire_logit = self.fire_head(h)          # (..., 1)
        fire_prob  = torch.sigmoid(fire_logit)
        return ctrl_out, fire_prob, fire_logit

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Rollout için: ctrl + Bernoulli(fire) sampling.

        Döndürür:
            action        : (..., 5)  — [ctrl4 | fire1]
            log_prob_fire : (..., 1)  — fire aksiyonu log-olasılığı
        """
        ctrl, fire_prob, fire_logit = self.forward(obs)
        dist = Bernoulli(logits=fire_logit)
        if deterministic:
            fire = (fire_prob >= 0.5).float()
        else:
            fire = dist.sample()
        log_prob_fire = dist.log_prob(fire)     # (..., 1)
        action = torch.cat([ctrl, fire], dim=-1)
        return action, log_prob_fire

    def action_for_grad(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Actor update için differentiable aksiyon.
        fire = sigmoid(logit)  — sample edilmez, gradient akabilir.

        Döndürür: (..., 5)
        """
        ctrl, fire_prob, _ = self.forward(obs)
        return torch.cat([ctrl, fire_prob], dim=-1)


class FACMACTwinCritic(nn.Module):
    """
    TD3 twin critic — Q-value aşırı tahmin sorununu minimize eder.
    İki bağımsız Q-ağı; target Q = min(Q1_tot, Q2_tot) ile actor güncellenir.

    Giriş  : [obs(obs_dim) | action(action_dim)]
    Çıkış  : (Q1, Q2) — her biri (...,) scalar
    """

    def __init__(self, obs_dim: int = 50, action_dim: int = 5, hidden: int = 256):
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
