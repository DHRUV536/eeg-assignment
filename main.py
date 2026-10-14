"""
Single entry point: `python main.py` reproduces every number and figure.

Stages (each writes to outputs/; trial data is cached in cache/):
  Task 1 / A1  data audit                 -> outputs/task1/audit.csv, audit_runs.csv
  Task 1 / A2  ERD and lateralisation     -> outputs/task1/a2_lateralisation.csv, figures/a2_*
  Task 1 / A3  confound decoders          -> outputs/task1/a3_*.csv, per_subject_table.csv, figures/a3_*
  Task 1 / B2  b2train, b25, b23, b24, b26 -> outputs/task1/b2_*.csv, figures/b2_*

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

    models = ["EEGNet", "TF-time", "TF-chan-id", "TF-chan-noid"]
    best_tf = "TF-time"   # best transformer by B2.2 mean accuracy (b2_2_tokenisation.csv)
    if want("b2train"):   # B2.1 scaling + B2.2 tokenisation (~5 h on 2 cores, resumable)
        from task1.b2_experiments import run_training, scaling_figure, summarise_runs, tokenisation_table
        run_training(out1, SEED)
        summarise_runs(out1); scaling_figure(out1); tokenisation_table(out1)
    if want("b25"):       # B2.5 first: B2.3 also re-tests its ocular-removed models
        from task1.b2_link import correlations, link_figure, removal_check, run_no_ocular_training
        correlations(out1, models)
        removal_check(out1, SEED)
        run_no_ocular_training(out1, SEED, ["EEGNet", best_tf])
        link_figure(out1, ["EEGNet", best_tf])
    if want("b23"):
        from task1.b2_displacement import (displacement_figure, first_layer_sensitivity, layer_similarity,
                                           pattern_change, run_displacement, run_displacement_no_ocular)
        run_displacement(out1, SEED)
        run_displacement_no_ocular(out1, SEED)
        first_layer_sensitivity(out1, SEED); layer_similarity(out1, SEED); pattern_change(out1)
        displacement_figure(out1)
    if want("b24"):
        from task1.b2_reliance import (map_to_part_a, reliance_figures, run_reliance, saliency_all_models,
                                       saliency_shares)
        saliency_all_models(out1, SEED, models)
        run_reliance(out1, SEED, best_tf)
        map_to_part_a(out1, models); saliency_shares(out1, models)
        reliance_figures(out1, best_tf)
    if want("b26"):
        from task1.b2_intervention import intervention_summary, run_intervention
        run_intervention(out1, SEED, ["EEGNet", best_tf])
        intervention_summary(out1)


if __name__ == "__main__":
    main()
