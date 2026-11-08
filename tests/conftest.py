"""
tests/conftest.py
=================
Pytest configuration.

The existing test scripts (test_aircraft_model.py, etc.) are standalone
runners that call sys.exit() at module level — incompatible with pytest
collection.  This conftest.py tells pytest to skip collecting them directly;
they are exercised instead through tests/test_all_scripts.py, which runs
each one as a subprocess.
"""

collect_ignore = [
    "test_aircraft_model.py",
    "test_env_step.py",
    "test_eval.py",
    "test_geometry_utils.py",
    "test_heuristic_agent.py",
    "test_normalization.py",
    "test_opponent_pool.py",
    "test_reward_model.py",
    "test_train_mappo.py",
    "test_trim_solver.py",
    "test_weapons_model.py",
]
