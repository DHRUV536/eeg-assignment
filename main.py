"""Reproduce every result with `python main.py` (no manual steps).

    python main.py               # all stages, in order
    python main.py --stages a1   # a subset
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

SEED = 42  # project seed; every stage derives its own seeds from this
ROOT = Path(__file__).resolve().parent
OUT_T1 = ROOT / "outputs" / "task1"


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def stage_a1() -> None:
    from task1 import data_audit
    data_audit.main(OUT_T1)


def stage_a2() -> None:
    from task1 import a2_erd
    a2_erd.main(OUT_T1)


# Order matters: later stages read audit.csv to know which subjects are kept.
STAGES = {
    "a1": stage_a1,
    "a2": stage_a2,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default=",".join(STAGES),
                    help=f"comma-separated subset of: {', '.join(STAGES)}")
    args = ap.parse_args()
    wanted = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in STAGES]
    if unknown:
        ap.error(f"unknown stage(s): {unknown}")

    seed_everything()
    for name in STAGES:  # run in canonical order regardless of how they were listed
        if name in wanted:
            print(f"=== {name} ===")
            STAGES[name]()


if __name__ == "__main__":
    main()
