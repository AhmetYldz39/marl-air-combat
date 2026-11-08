"""
aircombat_bench/envs/pettingzoo_env.py
=======================================
PettingZoo ParallelEnv adapter for DogfightEnv.

Wraps the native ``DogfightEnv`` without modifying it.  Existing trainers
that use the native interface continue to work unchanged.

Usage
-----
>>> from aircombat_bench.envs import DogfightParallelEnv
>>> env = DogfightParallelEnv.from_yaml("configs/config.yaml")
>>> obs, info = env.reset(seed=42)
>>> actions = {agent: env.action_space(agent).sample() for agent in env.agents}
>>> obs, rew, term, trunc, info = env.step(actions)
>>> env.close()

Conformance
-----------
Passes ``pettingzoo.test.parallel_api_test(env)``.

PettingZoo API contract (ParallelEnv, v1.27+)
----------------------------------------------
reset(seed, options) → (obs_dict, info_dict)
step(actions)        → (obs, rew, term, trunc, info)
observation_space(agent)
action_space(agent)
agents               : list of currently alive agents
possible_agents      : list of all agents (fixed)
max_num_agents       : int
num_agents           : int  (len(agents))
render()
close()
state()              : global state array (optional but present)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

try:
    from pettingzoo.utils.env import ParallelEnv as _PZParallelEnv
    import gymnasium as gym
    _HAS_PETTINGZOO = True
except ImportError:
    _HAS_PETTINGZOO = False
    _PZParallelEnv = object     # fallback so class definition succeeds

from aircombat_bench.envs.dogfight_env import DogfightEnv, BLUE, RED, ENV_VERSION
from aircombat_bench.envs.aircraft_model import ACTION_DIM, STATE_DIM


# ===========================================================================
# DogfightParallelEnv
# ===========================================================================

class DogfightParallelEnv(_PZParallelEnv):
    """PettingZoo ParallelEnv adapter for air combat simulation.

    Parameters
    ----------
    config : dict
        Full config as returned by ``yaml.safe_load``.
    curriculum_phase : int
        Curriculum phase for the native env (default 4 = 2v2 normal spawn).
    """

    # PettingZoo metadata
    metadata = {
        "render_modes": [],
        "name": "aircombat_bench_v1",
        "env_version": ENV_VERSION,
        "is_parallelizable": True,
    }

    def __init__(self, config: dict, curriculum_phase: int = 4):
        if not _HAS_PETTINGZOO:
            raise ImportError(
                "pettingzoo and gymnasium are required for DogfightParallelEnv. "
                "Install with: pip install pettingzoo gymnasium"
            )

        super().__init__()

        self._config     = config
        self._phase      = curriculum_phase
        self._native_env = DogfightEnv(config)
        self._native_env.set_curriculum_phase(curriculum_phase)

        # ── Ajan kimlikleri ───────────────────────────────────────────────
        # PettingZoo possible_agents değişmez; yalnızca Blue ajanları açık.
        # Red (heuristic) dahili olarak yönetilir.
        self.possible_agents: List[str] = list(self._native_env.blue_ids)
        self.agents: List[str]          = []   # reset() sonrası güncellenir

        # ── Gözlem ve aksiyon boyutları ───────────────────────────────────
        self._obs_dim    = self._native_env.obs_dim
        self._action_dim = self._native_env.action_dim   # 5

        # ── Observation / action space ─────────────────────────────────────
        # Her ajan için aynı box alanları.
        self._obs_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self._obs_dim,), dtype=np.float32
        )
        # Continuous 5D: [aileron, elevator, rudder, throttle, fire]
        # aileron/elevator/rudder ∈ [-1, 1], throttle ∈ [0, 1], fire ∈ {0, 1}
        low  = np.array([-1., -1., -1., 0.,  0.], dtype=np.float32)
        high = np.array([ 1.,  1.,  1., 1.,  1.], dtype=np.float32)
        self._act_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        # ── Heuristic rakip ───────────────────────────────────────────────
        from aircombat_bench.agents.heuristic_agent import MultiHeuristicPolicy
        _all_ids  = self._native_env.blue_ids + self._native_env.red_ids
        _team_map = {a: ("blue" if "blue" in a else "red") for a in _all_ids}
        self._heuristic = MultiHeuristicPolicy(config, _all_ids, _team_map)

        # Dahili durum
        self._obs_dict:  Dict[str, np.ndarray] = {}
        self._dones:     Dict[str, bool]        = {}
        self._red_ids    = list(self._native_env.red_ids)

    # -----------------------------------------------------------------------
    # PettingZoo API
    # -----------------------------------------------------------------------

    @property
    def unwrapped(self) -> "DogfightParallelEnv":
        return self

    @property
    def max_num_agents(self) -> int:
        return len(self.possible_agents)

    @property
    def num_agents(self) -> int:
        return len(self.agents)

    def observation_space(self, agent: str) -> "gym.spaces.Box":
        return self._obs_space

    def action_space(self, agent: str) -> "gym.spaces.Box":
        return self._act_space

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Ortamı sıfırla ve ilk gözlemleri döndür.

        Parameters
        ----------
        seed    : int | None — env.seed(seed) çağrılır (None ise atlanır)
        options : dict | None — şimdilik kullanılmıyor

        Returns
        -------
        observations : {agent_id: obs_array}
        infos        : {agent_id: {}}
        """
        if seed is not None:
            self._native_env.seed(seed)
        self._heuristic.reset()

        all_obs          = self._native_env.reset()
        self.agents      = list(self._native_env.blue_ids)
        self._obs_dict   = {a: all_obs[a] for a in self.agents}
        self._dones      = {a: False for a in self.agents}

        infos = {a: {} for a in self.agents}
        return dict(self._obs_dict), infos

    def step(
        self,
        actions: Dict[str, np.ndarray],
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Any],
    ]:
        """Bir adım ilerle.

        Parameters
        ----------
        actions : {blue_agent_id: 5D_action_array}
            Yalnızca mevcut Blue ajanların aksiyonları verilmeli.

        Returns
        -------
        observations : {agent_id: obs}
        rewards      : {agent_id: float}
        terminations : {agent_id: bool}  — ajan öldüğünde True
        truncations  : {agent_id: bool}  — yalnızca max_steps sonrası True
        infos        : {agent_id: dict}
        """
        # Ölü ajanlar için sıfır aksiyon
        full_actions = {}
        for aid in self._native_env.blue_ids:
            if aid in actions:
                full_actions[aid] = np.asarray(actions[aid], dtype=np.float32)
            else:
                full_actions[aid] = np.zeros(self._action_dim, dtype=np.float32)

        # Red heuristic aksiyonları
        states      = self._native_env.get_all_states()
        opp_raw     = self._heuristic.act(states)
        red_actions = {k: v for k, v in opp_raw.items() if k in set(self._red_ids)}
        full_actions.update(red_actions)

        obs_raw, rew_raw, done, info_raw = self._native_env.step(full_actions)

        # ── PettingZoo çıktı formatı ──────────────────────────────────────
        observations: Dict[str, np.ndarray] = {}
        rewards:      Dict[str, float]      = {}
        terminations: Dict[str, bool]       = {}
        truncations:  Dict[str, bool]       = {}
        infos:        Dict[str, Any]        = {}

        from aircombat_bench.envs.aircraft_model import STATE_ALIVE

        all_states = self._native_env.get_all_states()
        episode_done = done.get("__all__", False)
        is_truncated = (done.get("winner") == "draw") and episode_done

        new_agents = []
        for aid in self.possible_agents:
            s       = all_states.get(aid)
            is_dead = (s is None) or (s[STATE_ALIVE] < 0.5)

            observations[aid]  = obs_raw.get(
                aid, np.zeros(self._obs_dim, dtype=np.float32)
            )
            rewards[aid]       = float(rew_raw.get(aid, 0.0))
            terminations[aid]  = bool(is_dead and episode_done and not is_truncated)
            truncations[aid]   = bool(is_truncated)
            infos[aid]         = info_raw.get(aid, {})

            if not (terminations[aid] or truncations[aid]):
                new_agents.append(aid)

        # Ölü ajanları agents listesinden çıkar
        self.agents      = new_agents if not episode_done else []
        self._obs_dict   = observations
        return observations, rewards, terminations, truncations, infos

    def render(self) -> None:
        """Render desteği yok — hiçbir şey yapmaz."""

    def close(self) -> None:
        """Kaynakları serbest bırak (native env)."""
        pass   # DogfightEnv'in kapatılacak bir kaynağı yok

    def state(self) -> np.ndarray:
        """Tüm ajanların gözlemlerini birleştiren global durum vektörü.

        QMIX ve merkezi değerlendirici için kullanışlı.
        Boyut: n_agents × obs_dim.
        """
        parts = []
        for aid in self.possible_agents:
            obs = self._obs_dict.get(
                aid, np.zeros(self._obs_dim, dtype=np.float32)
            )
            parts.append(obs)
        return np.concatenate(parts, axis=0)

    # -----------------------------------------------------------------------
    # Native env erişimi
    # -----------------------------------------------------------------------

    @property
    def native(self) -> DogfightEnv:
        """Orijinal DogfightEnv örneğine doğrudan erişim."""
        return self._native_env

    # -----------------------------------------------------------------------
    # Yapıcı alternatifler
    # -----------------------------------------------------------------------

    @classmethod
    def from_yaml(
        cls,
        config_path,
        curriculum_phase: int = 4,
    ) -> "DogfightParallelEnv":
        """YAML config dosyasından ortam oluştur.

        Parameters
        ----------
        config_path    : str | pathlib.Path
        curriculum_phase : int (default 4 = 2v2 normal spawn)
        """
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return cls(config, curriculum_phase=curriculum_phase)


# ===========================================================================
# Scenario YAML loader
# ===========================================================================

#: Dahili senaryo dizini (paket içinde)
_SCENARIOS_DIR = Path(__file__).resolve().parent.parent / "scenarios"

#: Fallback: senaryo adından curriculum fazına eşleme
_SCENARIO_PHASE = {
    "1v1_close":   1,
    "1v1_dynamic": 2,
    "1v1_normal":  3,
    "2v2_normal":  4,
}


def load_scenario(scenario: str) -> dict:
    """Senaryo YAML dosyasını yükle ve döndür.

    Parameters
    ----------
    scenario : "1v1_close" | "1v1_dynamic" | "1v1_normal" | "2v2_normal"
        Senaryo adı — ``aircombat_bench/scenarios/<name>.yaml`` karşılığı.

    Returns
    -------
    dict
        Senaryo tanımı (schema_version, name, curriculum_phase, spawn, …).

    Raises
    ------
    ValueError
        Tanınmayan senaryo adı.
    FileNotFoundError
        YAML dosyası bulunamazsa.

    Examples
    --------
    >>> from aircombat_bench.envs import load_scenario
    >>> sc = load_scenario("2v2_normal")
    >>> sc["curriculum_phase"]
    4
    >>> sc["benchmark"]["is_headline"]
    True
    """
    if scenario not in _SCENARIO_PHASE:
        raise ValueError(
            f"Bilinmeyen senaryo: {scenario!r}. "
            f"Geçerliler: {sorted(_SCENARIO_PHASE)}"
        )

    yaml_path = _SCENARIOS_DIR / f"{scenario}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Senaryo YAML bulunamadı: {yaml_path}. "
            f"'pip install -e .' ile paketi kurulu mu?"
        )

    with open(yaml_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ===========================================================================
# make_env factory
# ===========================================================================

def make_env(
    scenario: str = "2v2_normal",
    config_path = None,
    **kwargs,
) -> "DogfightParallelEnv":
    """Senaryo adına göre DogfightParallelEnv oluştur.

    Senaryo tanımı ``aircombat_bench/scenarios/<name>.yaml`` dosyasından
    okunur.  ``curriculum_phase`` otomatik olarak ayarlanır — ek kod değişikliği
    gerekmez.

    Parameters
    ----------
    scenario    : "2v2_normal" | "1v1_close" | "1v1_dynamic" | "1v1_normal"
    config_path : YAML config yolu (None → repo kökündeki varsayılan)
    **kwargs    : DogfightParallelEnv'e aktarılan ek argümanlar

    Returns
    -------
    DogfightParallelEnv

    Examples
    --------
    >>> from aircombat_bench.envs import make_env
    >>> env = make_env("2v2_normal")
    >>> obs, info = env.reset(seed=42)
    >>> print(env.possible_agents)
    ['blue_0', 'blue_1']
    """
    if scenario not in _SCENARIO_PHASE:
        raise ValueError(
            f"Bilinmeyen senaryo: {scenario!r}. "
            f"Geçerliler: {sorted(_SCENARIO_PHASE)}"
        )

    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent.parent / "configs/config.yaml"

    # Senaryo YAML'ından curriculum fazını oku (YAML yoksa fallback kullan)
    try:
        sc = load_scenario(scenario)
        phase = int(sc["curriculum_phase"])
    except FileNotFoundError:
        phase = _SCENARIO_PHASE[scenario]

    env = DogfightParallelEnv.from_yaml(config_path, curriculum_phase=phase, **kwargs)
    return env
