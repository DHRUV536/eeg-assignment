"""
B2.6 - One intervention aimed at the mechanism found in B2.4/B2.5.

Mechanism (B2.5): with the ocular route removed, what is left to transfer
across subjects is mu/beta power at sensorimotor sites. Motor imagery
changes that power RELATIVE to each subject's own resting level (ERD), but
the network only sees the post-cue window, scaled by one number per trial.
So it has to read ERD from absolute power, and absolute power at each
electrode differs between people for reasons unrelated to imagery (skull
thickness, impedance, the undocumented reference).

Intervention: express every channel relative to its own pre-cue power, in
the same trial (see train.preprocess_variant 'no_ocular_erdnorm'):
    x~_c(t) = x_c(t) / sigma_c,       sigma_c = RMS of x_c over -1.5..-0.5 s
Any band power the network computes from x~_c is then
    log P~_c = log P_c(post) - log P_c(pre) = ERD_c (in log units),
which is exactly the quantity A2 shows lateralises with the imagined hand.
A per-subject, per-electrode gain g_c (anything multiplying the signal at
that electrode, the same before and after the cue) cancels:
    (g_c x_c) / (g_c sigma_c) = x_c / sigma_c,
so cross-subject differences of that kind no longer reach the network.
It uses only the trial itself, so no test-subject statistic is involved.
The baseline is band-limited to 8-30 Hz, so it carries power, not the
polarity of the slow eye-position signal that leaked the previous trial's
class in A3 (R5_precue_ocular).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from task1.train import N_FOLDS, SEEDS, Runner, load_net_data, make_folds


def run_intervention(out_dir: Path, seed: int, models):
    data = load_net_data(out_dir / "audit.csv", "no_ocular_erdnorm")
    folds = make_folds(np.unique(data["subject"]), seed)
    runner = Runner(out_dir / "b2_6_runs.csv", data, folds, seed)
    for m in models:
        for s in SEEDS:
            for f in range(N_FOLDS):
                runner.run("erd_norm", m, f, None, s)
