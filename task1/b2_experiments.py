"""
B2 experiments. Training jobs go through train.Runner (resumable, one CSV
row per test subject); analyses read the saved models and CSVs.

  B2.1 data scaling     EEGNet, TF-time x {5, 10, 20, all} subjects x 3 seeds x 5 folds
  B2.2 tokenisation     TF-time, TF-chan-id, TF-chan-noid at all subjects x 3 seeds x 5 folds
  B2.3 displacement     test-time only, on the B2.2/B2.1 "all" models
  B2.4 reliance         attention + gradient attribution, deletion tests
  B2.5 link to Part A   per-subject correlations; retraining without the ocular route
  B2.6 intervention     see b2_intervention.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from task1.plotting import C, MODEL_COLORS, plt, style
from task1.train import N_FOLDS, SEEDS, Runner, load_net_data, make_folds

ALL_MODELS = ["EEGNet", "TF-time", "TF-chan-id", "TF-chan-noid"]
SCALING_MODELS = ["EEGNet", "TF-time"]
SCALING_N = [5, 10, 20]


def setup(out_dir: Path, seed: int):
    data = load_net_data(out_dir / "audit.csv")
    folds = make_folds(np.unique(data["subject"]), seed)
    return data, folds


def run_training(out_dir: Path, seed: int):
    """B2.1 + B2.2 training. 'all' models first: B2.3/B2.4 need their weights."""
    data, folds = setup(out_dir, seed)
    runner = Runner(out_dir / "b2_runs.csv", data, folds, seed)
    for model in ALL_MODELS:
        for s in SEEDS:
            for f in range(N_FOLDS):
                runner.run("main", model, f, None, s, save_weights=True)
    for n in SCALING_N:
        for model in SCALING_MODELS:
            for s in SEEDS:
                for f in range(N_FOLDS):
                    runner.run("scaling", model, f, n, s)


# ---------------------------------------------------------------------------
# B2.1 / B2.2 summaries
# ---------------------------------------------------------------------------
def summarise_runs(out_dir: Path) -> pd.DataFrame:
    """One row per (model, n_train, seed): mean over the 37 test subjects.
    The 'main' runs are also the 'all subjects' point of the scaling curve."""
    runs = pd.read_csv(out_dir / "b2_runs.csv")
    runs["n_label"] = np.where(runs.experiment == "main", "all", runs.n_train_subjects.astype(str))
    per_seed = (runs.groupby(["model", "n_label", "seed"])
                .agg(mean_subject_acc=("acc", "mean"), n_subjects=("subject", "nunique"),
                     mean_train_subjects=("n_train_subjects", "mean"))
                .reset_index())
    per_seed.to_csv(out_dir / "b2_per_seed.csv", index=False, float_format="%.4f")
    return per_seed


def scaling_figure(out_dir: Path):
    style()
    ps = pd.read_csv(out_dir / "b2_per_seed.csv")
    order = ["5", "10", "20", "all"]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for m in SCALING_MODELS:
        sub = ps[ps.model == m]
        xs, means = [], []
        for i, n in enumerate(order):
            v = sub[sub.n_label == n].mean_subject_acc.values
            if len(v) == 0:
                continue
            x = sub[sub.n_label == n].mean_train_subjects.mean()
            ax.scatter(np.full(len(v), x), v, color=MODEL_COLORS[m], s=14, alpha=0.6, zorder=3)
            ax.vlines(x, v.min(), v.max(), color=MODEL_COLORS[m], lw=1, alpha=0.6)
            xs.append(x); means.append(v.mean())
        ax.plot(xs, means, color=MODEL_COLORS[m], marker="o", label=m, zorder=4)
        ax.text(xs[-1] + 0.6, means[-1], m, color=MODEL_COLORS[m], va="center", fontsize=8)
    ax.axhline(0.5, color=C["ink2"], lw=0.8, ls=":")
    ax.set_xlabel("training subjects")
    ax.set_ylabel("test accuracy (mean of 37 test subjects)")
    ax.set_title("B2.1 data scaling: each dot = one seed, line = mean of 3", loc="left")
    ax.legend(loc="lower right")
    fig.savefig(out_dir / "figures" / "b2_1_scaling.png")
    plt.close(fig)


def tokenisation_table(out_dir: Path) -> pd.DataFrame:
    ps = pd.read_csv(out_dir / "b2_per_seed.csv")
    t = ps[(ps.n_label == "all")].pivot(index="model", columns="seed", values="mean_subject_acc")
    t.columns = [f"seed{c}" for c in t.columns]
    t["mean"] = t.mean(axis=1); t["min"] = t.min(axis=1); t["max"] = t.max(axis=1)
    t = t.loc[[m for m in ALL_MODELS if m in t.index]]
    t.to_csv(out_dir / "b2_2_tokenisation.csv", float_format="%.4f")
    return t
