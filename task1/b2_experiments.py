"""B1 summary and B2.1 (data scaling) + B2.2 (tokenisation) runs and figures.

One fixed seed (42) per the assignment; every run is reported in b2_runs.csv.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from task1.models import EEGNet, EEGTransformer, n_params
from task1.train import OUT, SEED, run_all, subject_split

RUNS_CSV = OUT / "b2_runs.csv"
# The transformer used in B2.1 is the time-patch variant (decided before running, D27): it
# is the conventional EEG tokenisation and the cheapest of the three.
SCALING_TRANSFORMER = "time"
N_SUBSETS = [5, 20, None]                       # None = all training subjects


def planned_runs() -> list[dict]:
    train = subject_split()["train"]
    runs = []
    for n in N_SUBSETS:
        subs = train if n is None else train[:n]          # nested random subsets (train order)
        tag = "all" if n is None else str(n)
        runs.append(dict(run_id=f"eegnet_n{tag}", model_name="eegnet", tokeniser=None,
                         train_subjects=subs))
        runs.append(dict(run_id=f"tf-{SCALING_TRANSFORMER}_n{tag}", model_name="transformer",
                         tokeniser=SCALING_TRANSFORMER, train_subjects=subs))
    for tok in ["chan_id", "chan_noid"]:                  # B2.2, all training subjects
        runs.append(dict(run_id=f"tf-{tok}_nall", model_name="transformer", tokeniser=tok,
                         train_subjects=train))
    return runs


def posthoc_runs() -> list[dict]:
    """Post-hoc (added after seeing B2.2, D28): the pre-registered rule (stop on validation
    loss) kept the channel+identity model at epoch 1 because its validation loss rose from
    the start while validation accuracy improved. Re-run the three tokenisers stopping on
    validation accuracy, so the tokenisation comparison does not hinge on one stopping rule."""
    train = subject_split()["train"]
    return [dict(run_id=f"tf-{tok}_nall_stopacc", model_name="transformer", tokeniser=tok,
                 train_subjects=train, stop_on="acc") for tok in ["time", "chan_id", "chan_noid"]]


def b1_summary() -> pd.DataFrame:
    e = n_params(EEGNet())
    rows = [dict(model="EEGNet", tokeniser="", params=e, ratio_to_eegnet=1.0)]
    for tok in ["time", "chan_id", "chan_noid"]:
        p = n_params(EEGTransformer(tok))
        rows.append(dict(model="transformer", tokeniser=tok, params=p, ratio_to_eegnet=p / e))
    df = pd.DataFrame(rows)
    assert (df.ratio_to_eegnet <= 3).all(), "transformer exceeds 3x EEGNet"
    df.round(3).to_csv(OUT / "b1_params.csv", index=False)
    print(df.round(2).to_string(index=False))
    return df


def train_all() -> pd.DataFrame:
    budget = float(os.environ["EEG_TIME_BUDGET"]) if os.environ.get("EEG_TIME_BUDGET") else None
    return run_all(planned_runs() + posthoc_runs(), RUNS_CSV, budget)


def per_subject_long(runs: pd.DataFrame) -> pd.DataFrame:
    rows = [dict(run_id=r.run_id, subject=int(s), acc=a)
            for r in runs.itertuples() for s, a in json.loads(r.per_subject).items()]
    return pd.DataFrame(rows)


def common_mode_check() -> pd.DataFrame:
    """Post-hoc (D29), explains why channel tokens WITHOUT identity beat chance. That model is
    invariant to electrode order, so it can only use what is the same under any relabelling,
    e.g. the mean over all electrodes. If left/right differences were mirror-symmetric
    (F7 +a, F8 -a) that mean would carry nothing. Test it per subject, 0.2-1.0 s (saccade).
    Uses all kept subjects' data but no model and no labels beyond the class means."""
    from scipy.stats import ttest_rel
    from task1.train import network_data
    d = network_data()
    X, y, S, ch = d["X"], d["y"], d["subject"], list(d["ch_names"])
    t = np.arange(X.shape[-1]) / 160.0
    w = (t >= 0.2) & (t < 1.0)
    rows = []
    for s in np.unique(S):
        m = S == s
        diff = (X[m & (y == 0)][:, :, w].mean((0, 2)) - X[m & (y == 1)][:, :, w].mean((0, 2))) * 1e6
        rows.append(dict(subject=int(s), common_mode_LminusR_uV=float(diff.mean()),
                         F7_LminusR_uV=float(diff[ch.index("F7")]),
                         F8_LminusR_uV=float(diff[ch.index("F8")]),
                         n_channels_positive=int((diff > 0).sum())))
    df = pd.DataFrame(rows)
    df.round(3).to_csv(OUT / "b2_2_common_mode.csv", index=False)
    p = ttest_rel(df.common_mode_LminusR_uV, np.zeros(len(df))).pvalue
    print(f"[B2.2 post-hoc] mean over 64 channels, left-right, 0.2-1 s: "
          f"{df.common_mode_LminusR_uV.mean():+.2f} uV (t-test p={p:.1g}; "
          f"{int((df.common_mode_LminusR_uV > 0).sum())}/{len(df)} subjects > 0); "
          f"F7 {df.F7_LminusR_uV.mean():+.1f} uV vs F8 {df.F8_LminusR_uV.mean():+.1f} uV")
    return df


def report() -> None:
    runs = pd.read_csv(RUNS_CSV)
    if len(runs) < len(planned_runs()) + len(posthoc_runs()):
        raise RuntimeError(f"only {len(runs)} B2 runs finished; "
                           "run the b2train stage first")
    ps = per_subject_long(runs)
    ps.to_csv(OUT / "b2_per_subject.csv", index=False)
    n_all = len(subject_split()["train"])
    runs["n_label"] = runs.n_train_subjects.astype(str)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for name, model, tok, c in [("EEGNet", "eegnet", "", "tab:blue"),
                                (f"Transformer ({SCALING_TRANSFORMER})", "transformer",
                                 SCALING_TRANSFORMER, "tab:orange")]:
        d = runs[(runs.model == model) & (runs.tokeniser.fillna("") == tok)
                 & (runs.stop_on == "loss")].sort_values("n_train_subjects")
        ax[0].plot(d.n_train_subjects, d.test_acc, "o-", color=c, label=f"{name}: pooled test trials")
        for r in d.itertuples():
            s = ps[ps.run_id == r.run_id].acc
            ax[0].scatter(np.full(len(s), r.n_train_subjects) + (0.35 if model == "eegnet" else -0.35),
                          s, color=c, s=8, alpha=0.35)
    ax[0].axhline(0.5, color="k", lw=0.5)
    ax[0].set_xticks([5, 20, n_all])
    ax[0].set_xlabel("training subjects")
    ax[0].set_ylabel("test accuracy (10 held-out subjects)")
    ax[0].set_title("B2.1 data scaling (seed 42); dots = individual test subjects", fontsize=9)
    ax[0].legend(fontsize=8)

    toks = [("time-patch", "time"), ("channel +\nidentity", "chan_id"),
            ("channel,\nno identity", "chan_noid")]
    for k, (suffix, lab, col, dx) in enumerate([("", "stop on val loss (pre-registered)", "tab:orange", -0.2),
                                                ("_stopacc", "stop on val accuracy (post-hoc)", "tab:green", 0.2)]):
        for i, (_, tok) in enumerate(toks):
            rid = f"tf-{tok}_nall{suffix}"
            r = runs[runs.run_id == rid].iloc[0]
            s_ = ps[ps.run_id == rid].acc
            ax[1].bar(i + dx, r.test_acc, width=0.38, color=col, alpha=0.6, label=lab if i == 0 else None)
            ax[1].scatter(np.full(len(s_), i + dx), s_, color="k", s=6, zorder=3)
            ax[1].text(i + dx, 0.32, f"{r.test_acc:.3f}", ha="center", fontsize=7)
    ax[1].axhline(0.5, color="k", lw=0.5)
    ax[1].set_xticks(range(3), [t[0] for t in toks], fontsize=8)
    ax[1].set_ylim(0.3, 1.0)
    ax[1].set_ylabel("test accuracy")
    ax[1].legend(fontsize=7, loc="upper right")
    ax[1].set_title("B2.2 tokenisation (all 23 training subjects); dots = test subjects", fontsize=9)
    fig.savefig(OUT / "b2_scaling_tokenisation.png", dpi=120)
    plt.close(fig)
    common_mode_check()
    print(runs[["run_id", "stop_on", "n_params", "n_train_subjects", "epochs_run", "best_epoch",
                "val_acc", "test_acc", "test_acc_mean_over_subjects"]].round(3).to_string(index=False))
