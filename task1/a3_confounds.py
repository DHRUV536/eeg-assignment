"""A3: can left vs right be decoded from non-motor routes created by how trials were cued?

Cue (PhysioNet EEGMMIDB protocol, runs 4/8/12): a target appears on the LEFT or RIGHT of the
screen for ~4 s. So the label is confounded with stimulus position. Routes tested (specs fixed
in PREDICTIONS.md P1 before this ran):
  ocular        horizontal eye movement towards the target (frontal bipolar "HEOG" proxies)
  visual_evoked lateralised visual evoked response to the target's hemifield
  visual_alpha  posterior alpha lateralisation (attention to the hemifield)
  motor         positive control: sensorimotor mu/beta ERD (what *should* carry the label)

Outputs (outputs/task1/): a3_decoders.csv, per_subject_table.csv, a3_evidence.png
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.stats import binom, binomtest, spearmanr, wilcoxon
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from task1.a2_erd import BANDS, band_erd, tf_power
from task1.data import CACHE, kept_subjects, load_epochs

ROOT = Path(__file__).resolve().parents[1]
FEAT_CACHE = CACHE.parent / "a3_features"

# ---- decoder specifications (identical to PREDICTIONS.md P1) --------------------------
HEOG_PAIRS = [("F7", "F8"), ("AF7", "AF8"), ("Fp1", "Fp2")]
# 0.1-8 Hz: a saccade, and a gaze held on the target, are a step / sustained offset; a 1 Hz
# high-pass would turn them into a brief transient and throw most of that away.
OCULAR_BAND = (0.1, 8.0)
OCULAR_BINS = np.arange(0.0, 3.5 + 1e-9, 0.25)          # edges; 14 bins of 250 ms
EVOKED_PAIRS = [("PO7", "PO8"), ("PO3", "PO4"), ("O1", "O2")]
EVOKED_BINS = np.arange(0.05, 0.50 + 1e-9, 0.05)        # 9 bins of 50 ms (early VEP range)
POST_BASELINE = (-0.2, 0.0)                              # time-domain baseline just before cue
ALPHA_CH = ["PO7", "PO3", "O1", "PO8", "PO4", "O2"]
MOTOR_CH = ["C3", "C4", "C1", "C2", "C5", "C6", "FC3", "FC4", "CP3", "CP4"]
ROUTES = ["ocular", "visual_evoked", "visual_alpha", "motor"]

# ---- post-hoc check (added AFTER seeing the P1 results; not part of the prediction) ----
# The grand-average PO7-PO8 peak coincides with the saccade (~0.2 s), and left/right traces
# already differ before the cue: the zero-phase 1-40 Hz FIR spreads the ~45 uV saccade
# potential backwards in time. So visual_evoked may partly read eye movements. Test both
# routes in a window that ends before saccades start (0.05-0.15 s), on 0.1-40 Hz data whose
# high-pass is too slow to ring noticeably. If the visual route survives there and the
# ocular one does not, the visual effect is genuine.
EARLY_BAND = (0.1, 40.0)
EARLY_BINS = np.array([0.05, 0.10, 0.15])
POSTHOC_ROUTES = ["ocular_presaccade", "visual_presaccade"]
ALPHA_LEVEL = 0.05


def chance_threshold(n: int, alpha: float = ALPHA_LEVEL) -> float:
    """Smallest accuracy k/n with P(X >= k) <= alpha for X ~ Binomial(n, 0.5).

    Combrisson & Jerbi (2015). Assumes balanced classes and independent predictions; with
    leave-one-run-out CV each trial is predicted exactly once, by a model that never saw it.
    Close to but not exactly independent (the three fold models share training data), so this
    is a slightly lenient threshold; we also report group-level tests.
    """
    k = int(binom.ppf(1 - alpha, n, 0.5)) + 1
    while binom.sf(k - 1, n, 0.5) > alpha:   # guard against ppf rounding
        k += 1
    return k / n


def _binned_mean(x: np.ndarray, times: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """x: (trial, ch, time) -> (trial, ch * n_bins), mean amplitude per bin."""
    cols = [x[..., (times >= a) & (times < b)].mean(-1) for a, b in zip(edges[:-1], edges[1:])]
    return np.stack(cols, -1).reshape(len(x), -1)


def _bipolar(ep: mne.Epochs, pairs) -> np.ndarray:
    a = ep.get_data(picks=[p[0] for p in pairs])
    b = ep.get_data(picks=[p[1] for p in pairs])
    d = a - b
    base = (ep.times >= POST_BASELINE[0]) & (ep.times <= POST_BASELINE[1])
    return d - d[..., base].mean(-1, keepdims=True)


def subject_features(subject: int, use_cache: bool = True) -> dict:
    """All route features for one subject, plus labels and run ids (cached)."""
    path = FEAT_CACHE / f"S{subject:03d}.npz"
    if use_cache and path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}

    ep = load_epochs(subject)                                    # 1-40 Hz
    ep_lo = load_epochs(subject, l_freq=OCULAR_BAND[0], h_freq=OCULAR_BAND[1])
    # Same cue events, same window -> same trials; check rather than assume.
    assert np.array_equal(ep.events, ep_lo.events), f"trial mismatch S{subject}"

    feats = {
        "y": (ep.events[:, 2] == 2).astype(int),                 # 0 left, 1 right
        "run": ep.metadata["run"].to_numpy(),
        "ocular": _binned_mean(_bipolar(ep_lo, HEOG_PAIRS), ep_lo.times, OCULAR_BINS),
        "visual_evoked": _binned_mean(_bipolar(ep, EVOKED_PAIRS), ep.times, EVOKED_BINS),
    }
    ep_early = load_epochs(subject, l_freq=EARLY_BAND[0], h_freq=EARLY_BAND[1])
    assert np.array_equal(ep.events, ep_early.events)
    feats["ocular_presaccade"] = _binned_mean(_bipolar(ep_early, HEOG_PAIRS), ep_early.times, EARLY_BINS)
    feats["visual_presaccade"] = _binned_mean(_bipolar(ep_early, EVOKED_PAIRS), ep_early.times, EARLY_BINS)
    # Power features on CSD so each channel is local (no volume conduction of sensorimotor
    # mu into occipital channels or vice versa), reusing A2's exact TF and ERD definitions.
    csd = mne.preprocessing.compute_current_source_density(ep, verbose="error")
    p, times = tf_power(csd, ALPHA_CH + MOTOR_CH)
    pa, pm = p[:, : len(ALPHA_CH)], p[:, len(ALPHA_CH):]
    feats["visual_alpha"] = band_erd(pa, times, BANDS["mu"])    # 8-13 Hz = alpha band
    feats["motor"] = np.concatenate([band_erd(pm, times, BANDS["mu"]),
                                     band_erd(pm, times, BANDS["beta"])], axis=1)
    # Time courses kept only for the evidence figure.
    feats["heog_tc"] = _bipolar(ep_lo, HEOG_PAIRS[:1])[:, 0]     # F7-F8, (trial, time)
    feats["vep_tc"] = _bipolar(ep, EVOKED_PAIRS[:1])[:, 0]       # PO7-PO8
    feats["t_lo"], feats["t"] = ep_lo.times, ep.times

    FEAT_CACHE.mkdir(parents=True, exist_ok=True)
    tmp = FEAT_CACHE / f"tmp_{os.getpid()}_S{subject:03d}.npz"   # atomic write, see data.py
    np.savez(tmp, **feats)
    os.replace(tmp, path)
    return feats


def make_decoder():
    # Shrinkage LDA: with ~30 training trials and up to 42 features the plain covariance is
    # singular; Ledoit-Wolf shrinkage ("auto") regularises it without a tuned hyperparameter.
    return make_pipeline(StandardScaler(),
                         LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))


def within_subject_accuracy(X: np.ndarray, y: np.ndarray, runs: np.ndarray) -> float:
    """Leave-one-run-out: every trial predicted once by a model trained on the other runs.
    Not random k-fold: neighbouring trials in a run share slow drift, which would leak."""
    pred = np.empty_like(y)
    for tr, te in LeaveOneGroupOut().split(X, y, runs):
        pred[te] = make_decoder().fit(X[tr], y[tr]).predict(X[te])
    return float((pred == y).mean())


def cross_subject_accuracy(route: str, train_subjects, test_subjects) -> pd.DataFrame:
    """Train one decoder on the pooled train subjects, test on each test subject.
    The scaler is fit on training subjects only (Part B rule: no test-subject statistics).
    Used for write-up (a): how much a cross-subject model could score without imagery."""
    F = {s: subject_features(s) for s in set(train_subjects) | set(test_subjects)}
    Xtr = np.concatenate([F[s][route] for s in train_subjects])
    ytr = np.concatenate([F[s]["y"] for s in train_subjects])
    clf = make_decoder().fit(Xtr, ytr)
    rows = []
    for s in test_subjects:
        acc = float((clf.predict(F[s][route]) == F[s]["y"]).mean())
        rows.append(dict(subject=s, route=route, acc=acc, n=len(F[s]["y"]),
                         threshold=chance_threshold(len(F[s]["y"]))))
    return pd.DataFrame(rows)


def plot_evidence(F: dict, dec: pd.DataFrame, a2: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    for i, (key, tkey, title, unit_scale) in enumerate([
            ("heog_tc", "t_lo", "F7−F8 (horizontal-EOG proxy), 0.1–8 Hz", 1e6),
            ("vep_tc", "t", "PO7−PO8 (lateralised visual response), 1–40 Hz", 1e6)]):
        t = next(iter(F.values()))[tkey]
        m = (t >= -0.5) & (t <= 3.75)
        # Average within subject first, then across subjects (each subject weighs equally).
        L = np.array([f[key][f["y"] == 0].mean(0) for f in F.values()]) * unit_scale
        R = np.array([f[key][f["y"] == 1].mean(0) for f in F.values()]) * unit_scale
        for arr, lab, c in [(L, "left target", "tab:blue"), (R, "right target", "tab:red")]:
            mu, se = arr.mean(0), arr.std(0, ddof=1) / np.sqrt(len(arr))
            ax[i].plot(t[m], mu[m], color=c, label=lab)
            ax[i].fill_between(t[m], (mu - se)[m], (mu + se)[m], color=c, alpha=0.2)
        ax[i].axvline(0, color="k", lw=0.8)
        ax[i].axhline(0, color="k", lw=0.4)
        ax[i].set_title(f"Grand average ± SEM, {len(F)} subjects\n{title}", fontsize=9)
        ax[i].set_xlabel("time from cue (s)")
        ax[i].set_ylabel("µV")
        ax[i].legend(fontsize=8)

    # Per-subject accuracy by route, with that subject's chance threshold.
    present = set(a2.loc[a2.label == "present", "subject"])
    for j, route in enumerate(ROUTES):
        d = dec[dec.route == route]
        jit = (np.random.default_rng(j).random(len(d)) - 0.5) * 0.3
        col = ["tab:green" if s in present else "tab:grey" for s in d.subject]
        ax[2].scatter(j + jit, d.acc, c=col, s=14, zorder=3)
        ax[2].hlines(d.threshold.mean(), j - 0.3, j + 0.3, color="k", ls="--", lw=1)
        ax[2].text(j, 0.97, f"{int((d.acc >= d.threshold).sum())}/{len(d)}", ha="center", fontsize=8)
    ax[2].axhline(0.5, color="k", lw=0.5)
    ax[2].set_xticks(range(len(ROUTES)), ROUTES, fontsize=8)
    ax[2].set_ylim(0.2, 1.0)
    ax[2].set_ylabel("within-subject accuracy (LORO CV)")
    ax[2].set_title("Per subject; dashed = mean binomial threshold; numbers = above threshold\n"
                    "green = A2 mu-lateralised ERD present", fontsize=9)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(out_dir: Path) -> pd.DataFrame:
    out_dir = Path(out_dir)
    subjects = kept_subjects()
    F, rows = {}, []
    for i, s in enumerate(subjects, 1):
        F[s] = subject_features(s)
        n = len(F[s]["y"])
        for route in ROUTES:
            rows.append(dict(subject=s, route=route, n=n,
                             acc=within_subject_accuracy(F[s][route], F[s]["y"], F[s]["run"]),
                             threshold=chance_threshold(n)))
        print(f"[A3] {i:2d}/{len(subjects)} S{s:03d} done", flush=True)
        for route in POSTHOC_ROUTES:
            rows.append(dict(subject=s, route=route, n=n,
                             acc=within_subject_accuracy(F[s][route], F[s]["y"], F[s]["run"]),
                             threshold=chance_threshold(n)))
    dec = pd.DataFrame(rows)
    dec["posthoc"] = dec.route.isin(POSTHOC_ROUTES)
    dec["above_threshold"] = dec.acc >= dec.threshold
    dec.round(4).to_csv(out_dir / "a3_decoders.csv", index=False)

    # Group summary: how many subjects pass vs what chance alone would give.
    summ = []
    for route in ROUTES + POSTHOC_ROUTES:
        d = dec[dec.route == route]
        k = int(d.above_threshold.sum())
        summ.append(dict(route=route, posthoc=route in POSTHOC_ROUTES, mean_acc=d.acc.mean(), median_acc=d.acc.median(),
                         n_above=k, n_subjects=len(d),
                         expected_by_chance=round(ALPHA_LEVEL * len(d), 2),
                         p_count=binomtest(k, len(d), ALPHA_LEVEL, alternative="greater").pvalue,
                         p_wilcoxon_vs_50=wilcoxon(d.acc - 0.5, alternative="greater").pvalue))
    summ = pd.DataFrame(summ)
    summ.round(4).to_csv(out_dir / "a3_summary.csv", index=False)

    # Per-subject table required by the assignment: A2 mu LI + A3 accuracies with thresholds.
    a2 = pd.read_csv(out_dir / "a2_lateralisation.csv")
    wide = dec.pivot(index="subject", columns="route", values="acc").add_prefix("acc_")
    table = (a2[["subject", "n_left", "n_right", "li_mu_db", "es_mu", "p_mu", "q_mu", "label"]]
             .merge(wide.reset_index(), on="subject")
             .merge(dec.groupby("subject").threshold.first().rename("chance_threshold").reset_index(),
                    on="subject"))
    table.round(4).to_csv(out_dir / "per_subject_table.csv", index=False)

    rho = spearmanr(table.acc_ocular, table.acc_motor)
    rho_li = spearmanr(table.acc_ocular, table.li_mu_db)
    plot_evidence(F, dec, a2, out_dir / "a3_evidence.png")

    print(summ.round(3).to_string(index=False))
    print(f"[A3] Spearman ocular vs motor accuracy: rho={rho.statistic:+.2f} p={rho.pvalue:.3f}; "
          f"ocular accuracy vs mu LI: rho={rho_li.statistic:+.2f} p={rho_li.pvalue:.3f}")
    absent_ocular = table[(table.label == "absent") & (table.acc_ocular >= table.chance_threshold)]
    print(f"[A3] A2-absent subjects with ocular accuracy above threshold: "
          f"{absent_ocular.subject.tolist()}")
    return table


if __name__ == "__main__":
    main(ROOT / "outputs" / "task1")
