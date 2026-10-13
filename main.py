"""
Single entry point: `python main.py` reproduces every number and figure.

Currently runs:
  Task 1 / A1  data audit -> outputs/task1/audit.csv, audit_runs.csv
"""
import random
from pathlib import Path

import numpy as np
import torch

# One seed for the whole project, so any number can be traced to it.
# Later experiments derive their per-run seeds from this (documented in DECISIONS.md).
SEED = 42
OUT = Path(__file__).resolve().parent / "outputs"


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Deterministic kernels so a re-run gives the same numbers; some CPU ops
    # can still differ in the last bits, which is why we report seed spread.
    torch.use_deterministic_algorithms(True, warn_only=True)


def main() -> None:
    set_seeds(SEED)

    from audit import run_audit
    kept = run_audit(OUT / "task1")
    print(f"Task 1 subjects carried forward: {kept}")


if __name__ == "__main__":
    main()
