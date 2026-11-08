"""
train_facmac.py  (CLI shim)
===========================
Backward-compatible entry point — delegates to aircombat_bench.algorithms.facmac.

All logic lives in:  aircombat_bench/algorithms/facmac.py
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aircombat_bench.algorithms.facmac import main  # noqa: F401

if __name__ == "__main__":
    main()
