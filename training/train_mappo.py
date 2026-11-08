"""
train_mappo.py  (CLI shim)
==========================
Backward-compatible entry point — delegates to aircombat_bench.algorithms.mappo.

All logic lives in:  aircombat_bench/algorithms/mappo.py
"""
import sys
from pathlib import Path

# Ensure repo root is on sys.path (needed when running without `pip install -e .`)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aircombat_bench.algorithms.mappo import main, MAPPOActor, MAPPOCritic, GATMAPPOActor, GATMAPPOCritic  # noqa: F401

if __name__ == "__main__":
    main()
