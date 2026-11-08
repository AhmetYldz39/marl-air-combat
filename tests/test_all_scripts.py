"""
test_all_scripts.py
===================
Pytest-compatible wrapper that runs each standalone test script as a
subprocess and asserts exit code 0.

This lets pytest drive CI without modifying the original test files.
Running directly still works:  python tests/test_aircraft_model.py
"""
import subprocess
import sys
import pytest
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).parent

# Ordered: fast / no-network first, slow / integration last.
STANDALONE_SCRIPTS = [
    "test_geometry_utils.py",
    "test_aircraft_model.py",
    "test_weapons_model.py",
    "test_trim_solver.py",
    "test_normalization.py",
    "test_reward_model.py",
    "test_heuristic_agent.py",
    "test_env_step.py",
    "test_opponent_pool.py",
    "test_train_mappo.py",
    "test_eval.py",
]


@pytest.mark.parametrize("script", STANDALONE_SCRIPTS)
def test_script(script: str) -> None:
    """Run *script* as a subprocess and assert it exits with code 0."""
    path = _SCRIPTS_DIR / script
    result = subprocess.run(
        [sys.executable, "-X", "utf8", str(path)],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    if result.returncode != 0:
        # Show the tail of stdout so failures are readable in CI logs.
        tail = (result.stdout + result.stderr)[-3000:]
        pytest.fail(f"{script} exited with code {result.returncode}:\n{tail}")
