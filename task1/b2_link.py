"""
B2.5 - Link to Part A, and retraining with the dominant confound removed.

1. Per test subject: network accuracy (mean over 3 seeds) vs the A2
   lateralisation index (mu, beta) and the A3 decoders (ocular, visual
   evoked, pre-cue ocular, motor). Spearman correlations (n = 37), plus a
   linear model with all predictors together to separate overlapping ones
   (ocular and motor accuracies are themselves correlated across subjects).
2. Removal check: before retraining, the same within-subject decoders as A3
   are run on the network INPUTS with and without the ocular route, to show
   what the removal actually takes out and what it leaves.
3. Retraining: EEGNet and the best transformer on the no_ocular inputs,
   same folds, seeds and training length as the main runs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter, sosfiltfilt

from task1.a3_confounds import POSTERIOR, cv_accuracy
from task1.a2_erd import TF_CHANNELS
from task1.models import SFREQ
from task1.plotting import C, MODEL_COLORS, plt, style
from task1.train import N_FOLDS, SEEDS, OCULAR_SITES, Runner, load_net_data, make_folds

PREDICTORS = ["LI_mu_db", "LI_beta_db", "acc_R1_ocular", "acc_R2_visual_evoked",
              "acc_R5_precue_ocular", "acc_motor_sensorimotor"]


def subject_accuracy(runs_csv: Path, experiment="main") -> pd.DataFrame:
    r = pd.read_csv(runs_csv)
    r = r[r.experiment == experiment]
    return r.groupby(["model", "subject"]).acc.mean().unstack(0)


def correlations(out_dir: Path, models) -> pd.DataFrame:
    acc = subject_accuracy(out_dir / "b2_runs.csv")
    table = pd.read_csv(out_dir / "per_subject_table.csv").set_index("subject")
    rows = []
    for m in models:
        for p in PREDICTORS:
            r = stats.spearmanr(acc[m], table.loc[acc.index, p])
            rows.append({"model": m, "predictor": p, "spearman_rho": r.statistic, "p": r.pvalue, "n": len(acc)})
        # All predictors together (z-scored): which ones carry unique information?
        Z = table.loc[acc.index, PREDICTORS]
        Z = (Z - Z.mean()) / Z.std()
        A = np.c_[np.ones(len(Z)), Z.values]
        yv = acc[m].values
        beta, *_ = np.linalg.lstsq(A, yv, rcond=None)
        resid = yv - A @ beta
        dof = len(yv) - A.shape[1]
        se = np.sqrt(np.diag(np.linalg.inv(A.T @ A)) * (resid @ resid) / dof)
        for i, p in enumerate(PREDICTORS):
            rows.append({"model": m, "predictor": f"joint:{p}", "beta_per_sd": beta[i + 1],
                         "p": 2 * stats.t.sf(abs(beta[i + 1] / se[i + 1]), dof), "n": len(yv)})
        rows.append({"model": m, "predictor": "joint:R2",
                     "beta_per_sd": 1 - (resid @ resid) / ((yv - yv.mean()) @ (yv - yv.mean()))})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "b2_5_correlations.csv", index=False, float_format="%.4g")
    return df


# ---------------------------------------------------------------------------
# What does the removal take out? Within-subject decoders on network inputs.
# ---------------------------------------------------------------------------
def _input_routes(X, ch_names):
    ch = list(ch_names)
    t = np.arange(X.shape[-1]) / SFREQ
    out = {}
    # slow time-locked amplitude, all channels (eye movement / eye position)
    # 1 s bins keep this at 4 x 64 features: the saccade and the eye-position
    # shift last ~1 s, and 512 features made the shrinkage LDA very slow.
    edges = np.arange(0, 4.0001, 1.0)
    out["slow_amplitude_all_ch"] = np.concatenate(
        [X[..., (t >= a) & (t < b)].mean(-1) for a, b in zip(edges[:-1], edges[1:])], axis=1)
    # early posterior evoked response
    post = [ch.index(c) for c in POSTERIOR]
    edges = np.arange(0.05, 0.5001, 0.05)
    out["posterior_evoked"] = np.concatenate(
        [X[:, post][..., (t >= a) & (t < b)].mean(-1) for a, b in zip(edges[:-1], edges[1:])], axis=1)
    # mu/beta log power at the sensorimotor channels
    idx = [ch.index(c) for c in TF_CHANNELS]
    feats = []
    for lo, hi in ((8, 13), (13, 30)):
        sos = butter(4, [lo, hi], btype="bandpass", fs=SFREQ, output="sos")
        feats.append(np.log((sosfiltfilt(sos, X[:, idx], axis=-1) ** 2).mean(-1) + 1e-12))
    out["mu_beta_power_sensorimotor"] = np.concatenate(feats, axis=1)
    return out


def removal_check(out_dir: Path, seed: int) -> pd.DataFrame:
    rows = []
    for variant in ("default", "no_ocular"):
        d = load_net_data(out_dir / "audit.csv", variant)
        for s in np.unique(d["subject"]):
            k = d["subject"] == s
            feats = _input_routes(d["X"][k], d["ch_names"])
            for name, F in feats.items():
                F = np.nan_to_num(F)
                rows.append({"variant": variant, "subject": int(s), "route": name,
                             "acc": cv_accuracy(F, d["y"][k], seed + int(s), n_repeats=3)})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "b2_5_removal_check.csv", index=False, float_format="%.4f")
    return df


def run_no_ocular_training(out_dir: Path, seed: int, models):
    data = load_net_data(out_dir / "audit.csv", "no_ocular")
    folds = make_folds(np.unique(data["subject"]), seed)
    runner = Runner(out_dir / "b2_5_runs.csv", data, folds, seed)
    for m in models:
        for s in SEEDS:
            for f in range(N_FOLDS):
                # weights kept: B2.3 re-tests displacement on these motor-only models
                runner.run("no_ocular", m, f, None, s, tag="", save_weights=True)


def link_figure(out_dir: Path, models):
    style()
    acc = subject_accuracy(out_dir / "b2_runs.csv")
    acc_no = subject_accuracy(out_dir / "b2_5_runs.csv", "no_ocular") if (out_dir / "b2_5_runs.csv").exists() else None
    table = pd.read_csv(out_dir / "per_subject_table.csv").set_index("subject").loc[acc.index]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4))
    for ax, (p, lab) in zip(axes[:2], (("acc_R1_ocular", "A3 ocular decoder accuracy"),
                                        ("LI_mu_db", "A2 μ lateralisation index (dB)"))):
        for m in models:
            r = stats.spearmanr(acc[m], table[p]).statistic
            ax.scatter(table[p], acc[m], s=16, color=MODEL_COLORS[m], edgecolor="white", lw=0.6,
                       label=f"{m} (ρ = {r:+.2f})", zorder=3)
        if p == "LI_mu_db" and acc_no is not None:
            for m in models:
                if m in acc_no:
                    r = stats.spearmanr(acc_no[m], table.loc[acc_no.index, p]).statistic
                    ax.scatter(table.loc[acc_no.index, p], acc_no[m], s=18, facecolor="none",
                               edgecolor=MODEL_COLORS[m], marker="s", lw=1, zorder=3,
                               label=f"{m}, ocular removed (ρ = {r:+.2f})")
        ax.set_xlabel(lab); ax.set_ylabel("network test accuracy (mean of 3 seeds)")
        ax.axhline(0.5, color=C["ink2"], lw=0.8, ls=":")
        ax.legend(fontsize=7, loc="upper right" if p == "LI_mu_db" else "lower right")
    axes[0].set_title("Per test subject: network vs Part A", loc="left")
    ax = axes[2]
    if acc_no is not None:
        for i, m in enumerate(models):
            if m not in acc_no:
                continue
            a, b = acc[m], acc_no[m]
            ax.plot([i - 0.15, i + 0.15], [a.mean(), b.mean()], color=MODEL_COLORS[m], lw=2, marker="o")
            for sj in acc.index:
                ax.plot([i - 0.15, i + 0.15], [a[sj], b[sj]], color=MODEL_COLORS[m], lw=0.5, alpha=0.3)
        ax.set_xticks(range(len(models))); ax.set_xticklabels(models)
        ax.axhline(0.5, color=C["ink2"], lw=0.8, ls=":")
        ax.set_ylabel("test accuracy (thin = one subject)")
        ax.set_title("All inputs (left) vs ocular route removed (right)", loc="left", fontsize=8)
    fig.savefig(out_dir / "figures" / "b2_5_link.png")
    plt.close(fig)
