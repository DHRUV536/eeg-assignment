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
    return run_all(planned_runs(), RUNS_CSV, budget)


def per_subject_long(runs: pd.DataFrame) -> pd.DataFrame:
    rows = [dict(run_id=r.run_id, subject=int(s), acc=a)
            for r in runs.itertuples() for s, a in json.loads(r.per_subject).items()]
    return pd.DataFrame(rows)


def report() -> None:
    runs = pd.read_csv(RUNS_CSV)
    if len(runs) < len(planned_runs()):
        raise RuntimeError(f"only {len(runs)}/{len(planned_runs())} B2 runs finished; "
                           "run the b2train stage first")
    ps = per_subject_long(runs)
    ps.to_csv(OUT / "b2_per_subject.csv", index=False)
    n_all = len(subject_split()["train"])
    runs["n_label"] = runs.n_train_subjects.astype(str)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for name, model, tok, c in [("EEGNet", "eegnet", "", "tab:blue"),
                                (f"Transformer ({SCALING_TRANSFORMER})", "transformer",
                                 SCALING_TRANSFORMER, "tab:orange")]:
        d = runs[(runs.model == model) & (runs.tokeniser.fillna("") == tok)].sort_values("n_train_subjects")
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

    tok_runs = [("time-patch", f"tf-time_nall"), ("channel + identity", "tf-chan_id_nall"),
                ("channel, no identity", "tf-chan_noid_nall")]
    for i, (lab, rid) in enumerate(tok_runs):
        r = runs[runs.run_id == rid].iloc[0]
        s = ps[ps.run_id == rid].acc
        ax[1].bar(i, r.test_acc, color="tab:orange", alpha=0.6)
        ax[1].scatter(np.full(len(s), i), s, color="k", s=10, zorder=3)
        ax[1].text(i, r.test_acc + 0.01, f"{r.test_acc:.3f}", ha="center", fontsize=8)
    ax[1].axhline(0.5, color="k", lw=0.5)
    ax[1].set_xticks(range(3), [t[0] for t in tok_runs], fontsize=8)
    ax[1].set_ylim(0.3, 1.0)
    ax[1].set_ylabel("test accuracy")
    ax[1].set_title("B2.2 tokenisation (all training subjects); dots = test subjects", fontsize=9)
    fig.savefig(OUT / "b2_scaling_tokenisation.png", dpi=120)
    plt.close(fig)
    print(runs[["run_id", "n_params", "n_train_subjects", "epochs_run", "best_epoch",
                "val_acc", "test_acc", "test_acc_mean_over_subjects"]].round(3).to_string(index=False))
