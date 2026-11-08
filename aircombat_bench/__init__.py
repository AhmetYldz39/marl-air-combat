"""
aircombat-bench
===============
Multi-agent reinforcement learning benchmark for 3-D air combat
with 6-DoF F-16 dynamics.

Five-line quickstart::

    from aircombat_bench.envs import make_env
    from aircombat_bench.zoo import load_baseline

    policy = load_baseline("facmac_omr")      # best baseline (Win=84.2%)
    env = make_env("2v2_normal")              # headline benchmark scenario
    obs, _ = env.reset(seed=42)
    while env.agents:
        actions = policy.act(obs, env.agents)
        obs, rew, term, trunc, info = env.step(actions)

Baselines are available through :mod:`aircombat_bench.zoo`.
Scenarios and PettingZoo API: :mod:`aircombat_bench.envs`.
Eval runner: :mod:`aircombat_bench.eval.runner`.
"""

__version__ = "0.1.0"
__author__  = "Ahmet Yıldız"
__license__ = "MIT"

# Public top-level re-exports for convenience
from aircombat_bench.envs.dogfight_env import DogfightEnv   # noqa: F401
from aircombat_bench.envs.pettingzoo_env import (           # noqa: F401
    make_env,
    load_scenario,
)
from aircombat_bench.zoo.loader import (                    # noqa: F401
    load_baseline,
    manifest,
)
