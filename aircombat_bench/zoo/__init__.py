"""Model zoo: load pre-trained baselines from checkpoint files.

Quickstart (five lines)
-----------------------
>>> from aircombat_bench.envs import make_env
>>> from aircombat_bench.zoo import load_baseline
>>> policy = load_baseline("facmac_omr")
>>> env = make_env("2v2_normal")
>>> obs, _ = env.reset(seed=42)
>>> while env.agents:
...     actions = policy.act(obs, env.agents)
...     obs, rew, term, trunc, info = env.step(actions)

Available baselines
-------------------
  mappo        — MAPPO baseline               Win=46.4%
  mappo_gat    — MAPPO+GAT                    Win=69.8%
  mappo_gat_om — MAPPO+GAT+OM+Role            Win=70.8%
  qmix         — QMIX (162 discrete actions)  Win=65.0%
  facmac       — Pure FACMAC                  Win=82.0%
  facmac_omr   — FACMAC+OM+REINFORCE (best)   Win=84.2%

All figures: 500 ep, seed=42, deterministic, ENV_VERSION=1.0.0.
"""

from aircombat_bench.zoo.loader import load_baseline, manifest, BaselinePolicy  # noqa: F401
