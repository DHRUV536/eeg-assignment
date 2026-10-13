"""
Single entry point: `python main.py` reproduces every number and figure.

Stages (each writes to outputs/; trial data is cached in cache/):
  Task 1 / A1  data audit                 -> outputs/task1/audit.csv, audit_runs.csv
  Task 1 / A2  ERD and lateralisation     -> outputs/task1/a2_lateralisation.csv, figures/a2_*
  Task 1 / A3  confound decoders          -> outputs/task1/a3_*.csv, per_subject_table.csv, figures/a3_*
  (later stages are added as they are implemented)

Data: EDF files are read from ../files (or $EEGBCI_DIR) and downloaded through
MNE only if missing.
"""
import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch

# One seed for the whole project; every stage derives its own seeds from it
# with a fixed offset (DECISIONS.md D0).
SEED = 42
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Deterministic kernels so a re-run gives the same numbers; some CPU ops
    # can still differ in the last bits across machines, hence seed spreads.
    torch.use_deterministic_algorithms(True, warn_only=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", default="all",
                        help="comma-separated subset, e.g. a1,a2 (default: all)")
    args = parser.parse_args()
    stages = None if args.stages == "all" else set(args.stages.split(","))
    want = lambda name: stages is None or name in stages

    os.chdir(ROOT)                          # relative paths inside stages resolve here
    set_seeds(SEED)
    out1 = OUT / "task1"

    if want("a1"):
        from task1.audit import run_audit
        kept = run_audit(out1)
        print(f"A1: {len(kept)} subjects carried forward")
    if want("a2"):
        from task1.a2_erd import run_a2
        run_a2(out1, seed=SEED)
    if want("a3"):
        from task1.a3_confounds import per_subject_table, run_a3
        run_a3(out1, seed=SEED)
        per_subject_table(out1)


if __name__ == "__main__":
    main()
