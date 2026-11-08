"""
rollout_buffer.py
=================
RolloutBuffer — PyTorch bağımlılığı olmayan saf numpy buffer.
train_mappo.py tarafından import edilir.
"""
import numpy as np


class RolloutBuffer:
    """
    n_steps adımlık rollout verisi — tüm eğitilen ajanlar için.

    Her ajan için ayrı buffer tutulur, update sırasında birleştirilir.
    """

    def __init__(self, n_steps: int, n_agents: int,
                 obs_dim: int, action_dim: int, global_obs_dim: int):
        self.n_steps        = n_steps
        self.n_agents       = n_agents
        self.obs_dim        = obs_dim
        self.action_dim     = action_dim
        self.global_obs_dim = global_obs_dim
        self.reset()

    def reset(self):
        n, a = self.n_steps, self.n_agents
        self.obs        = np.zeros((n, a, self.obs_dim),        dtype=np.float32)
        self.actions    = np.zeros((n, a, self.action_dim),     dtype=np.float32)
        self.log_probs  = np.zeros((n, a),                      dtype=np.float32)
        self.rewards    = np.zeros((n, a),                      dtype=np.float32)
        self.dones      = np.zeros((n, a),                      dtype=np.float32)
        self.values     = np.zeros((n, a),                      dtype=np.float32)
        self.global_obs = np.zeros((n, self.global_obs_dim),    dtype=np.float32)
        self._ptr       = 0

    def add(self, obs: dict, actions: dict, log_probs: dict,
            rewards: dict, dones: dict, values: np.ndarray,
            global_obs: np.ndarray, agent_ids: list):
        """Bir adımın verisini buffer'a ekle."""
        assert self._ptr < self.n_steps, "Buffer dolu — önce get() çağırın."
        t = self._ptr
        for i, aid in enumerate(agent_ids):
            self.obs[t, i]      = obs[aid]
            self.actions[t, i]  = actions[aid]
            self.log_probs[t, i]= log_probs[aid]
            self.rewards[t, i]  = rewards[aid]
            self.dones[t, i]    = float(dones.get(aid, False))
            self.values[t, i]   = values[i]
        self.global_obs[t] = global_obs
        self._ptr += 1

    def compute_gae(self, last_values: np.ndarray,
                    gamma: float, gae_lambda: float) -> tuple:
        """
        GAE (Generalized Advantage Estimation).

        Parameters
        ----------
        last_values : (n_agents,) — bootstrap için son adım value tahmini
        gamma       : discount faktörü
        gae_lambda  : GAE lambda

        Returns
        -------
        advantages : (n_steps, n_agents)
        returns    : (n_steps, n_agents)  — value target
        """
        T = self.n_steps
        advantages = np.zeros_like(self.rewards)
        last_gae   = np.zeros(self.n_agents, dtype=np.float32)

        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_values       = last_values
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_values       = self.values[t + 1]

            delta    = (self.rewards[t]
                        + gamma * next_values * next_non_terminal
                        - self.values[t])
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + self.values
        return advantages, returns

    def get(self) -> dict:
        """Buffer içeriğini numpy dict olarak döndürür."""
        assert self._ptr == self.n_steps, "Buffer henüz dolmadı."
        return {
            "obs":        self.obs,
            "actions":    self.actions,
            "log_probs":  self.log_probs,
            "global_obs": self.global_obs,
        }
