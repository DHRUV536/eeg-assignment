"""
A3 - Confound audit: can left vs right be decoded from non-motor routes?

Each route (PREDICTIONS.md, A3) gets a decoder restricted to the channels,
band and time window where that route, and as little else as possible, lives.
All decoders share one classifier (standardise -> shrinkage LDA) and one
cross-validation scheme, so their accuracies are directly comparable.

Chance threshold for a subject with n trials: the smallest accuracy k/n such
that P(X >= k) <= 0.05 for X ~ Binomial(n, 0.5) (Combrisson & Jerbi 2015).
Classes are balanced to within 1-3 trials per subject, so p = 0.5 is the
right null. Because CV folds are not independent tests, the threshold's real
false-positive rate is checked empirically: every decoder is also run once
with shuffled labels and the fraction of subjects crossing the threshold is
reported (should be about 5%).
"""
from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from task1.a2_erd import TF_CHANNELS, apply_reference
from task1.audit import IMAGERY_RUNS, fetch_run_paths
from task1.data import SFREQ, kept_subjects, load_subject
from task1.plotting import C, plt, style

ALPHA = 0.05
N_SPLITS, N_REPEATS = 5, 10

OCULAR = ["Fp1", "Fp2", "AF7", "AF8", "F7", "F8", "FT7", "FT8"]
POSTERIOR = ["P7", "P8", "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]
PERIPHERAL = ["FT7", "FT8", "T7", "T8", "T9", "T10", "TP7", "TP8", "F7", "F8", "P7", "P8", "O1", "O2", "Iz"]

# One entry per route. kind "erp": mean amplitude in time bins of the
# low-passed signal (time-locked activity); kind "power": log band power in
# the window, relative to the pre-cue baseline when one is given.
ROUTES = {
    # Raw reference: EOG is large and the LDA can form any bipolar
    # combination (e.g. F7-F8) itself; no spatial filter is needed.
    "R1_ocular": dict(ref="raw", channels=OCULAR, kind="erp", lowpass=4.0,
                      window=(0.0, 3.975), bin_s=0.5, baseline=(-0.5, 0.0)),
    # CSD keeps the posterior signal local, so frontal EOG cannot leak in.
    "R2_visual_evoked": dict(ref="csd", channels=POSTERIOR, kind="erp", lowpass=20.0,
                             window=(0.05, 0.5), bin_s=0.05, baseline=(-0.2, 0.0)),
    "R3_posterior_alpha": dict(ref="csd", channels=POSTERIOR, kind="power", bands=[(8, 13)],
                               window=(0.5, 3.975), baseline=(-1.5, -0.5)),
    # 30-78 Hz with 57-63 Hz skipped, so the 60 Hz notch region adds nothing.
    "R4_emg": dict(ref="csd", channels=PERIPHERAL, kind="power", bands=[(30, 57), (63, 78)],
                   window=(0.5, 3.975), baseline=(-1.5, -0.5)),
    # Before the cue nothing about the upcoming class should be in the signal.
    "R5_precue": dict(ref="csd", channels="all", kind="power", bands=[(8, 13), (13, 30)],
                      separate_bands=True, window=(-2.0, -0.25), baseline=None),
    # Added AFTER the first run, when a3_mechanisms.png showed a ~15 uV F7-F8
    # class difference before the cue. Combined with the strongly alternating
    # class sequence (a3_cue_timing_and_sequence.csv), eye position left over
    # from the previous trial could predict the next class. The first pre-cue
    # test used mu/beta power and could not see this; this one uses the same
    # low-frequency ocular signal as R1, before the cue, with no baseline.
    "R5_precue_ocular": dict(ref="raw", channels=OCULAR, kind="erp", lowpass=4.0,
                             window=(-2.0, -0.25), bin_s=0.25, baseline=None),
    # Not a confound route: the motor reference the others are compared with.
    "motor_sensorimotor": dict(ref="csd", channels=TF_CHANNELS, kind="power",
                               bands=[(8, 13), (13, 30)], separate_bands=True,
                               window=(0.5, 3.975), baseline=(-1.5, -0.5)),
}


def chance_threshold(n: int, alpha: float = ALPHA) -> float:
    """Smallest accuracy k/n with P(Binomial(n, .5) >= k) <= alpha."""
    k = np.arange(n + 1)
    ok = stats.binom.sf(k - 1, n, 0.5) <= alpha
    return k[ok][0] / n


def _filt(x, lo, hi):
    if lo is None:
        sos = butter(4, hi, btype="lowpass", fs=SFREQ, output="sos")
    else:
        sos = butter(4, [lo, hi], btype="bandpass", fs=SFREQ, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def route_features(d: dict, route: dict, Xref: dict) -> np.ndarray:
    ch = list(d["ch_names"])
    X = Xref[route["ref"]]
    idx = list(range(len(ch))) if route["channels"] == "all" else [ch.index(c) for c in route["channels"]]
    X = X[:, idx]
    t = d["times"]
    win = (t >= route["window"][0]) & (t <= route["window"][1])
    if route["kind"] == "erp":
        x = _filt(X, None, route["lowpass"])
        if route["baseline"] is not None:
            b = (t >= route["baseline"][0]) & (t <= route["baseline"][1])
            x = x - x[..., b].mean(-1, keepdims=True)
        edges = np.arange(route["window"][0], route["window"][1] + 1e-9, route["bin_s"])
        feats = [x[..., (t >= lo) & (t < hi)].mean(-1) for lo, hi in zip(edges[:-1], edges[1:])]
        return np.concatenate(feats, axis=1)
    # power
    per_band = []
    for lo, hi in route["bands"]:
        p = _filt(X, lo, hi) ** 2
        per_band.append(p)
    if not route.get("separate_bands"):
        per_band = [np.mean(per_band, axis=0)]
    feats = []
    for p in per_band:
        f = np.log(p[..., win].mean(-1))
        if route["baseline"] is not None:
            b = (t >= route["baseline"][0]) & (t <= route["baseline"][1])
            f = f - np.log(p[..., b].mean(-1))
        feats.append(f)
    return np.concatenate(feats, axis=1)


def cv_accuracy(F: np.ndarray, y: np.ndarray, seed: int) -> float:
    clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=seed)
    correct = np.zeros(N_REPEATS)
    for i, (tr, te) in enumerate(cv.split(F, y)):
        clf.fit(F[tr], y[tr])
        correct[i // N_SPLITS] += (clf.predict(F[te]) == y[te]).sum()
    # Every trial is tested once per repeat; average the repeats.
    return float((correct / len(y)).mean())


def order_features(d: dict) -> np.ndarray:
    """R5b: the class from nothing but the trial's position (run, index in run)."""
    return np.column_stack([d["trial_in_run"], d["run"]]).astype(float)


def cue_timing_check(subjects) -> pd.DataFrame:
    """R6 and the label sequence, straight from the annotations."""
    rows = []
    for s in subjects:
        durs = {0: [], 1: []}
        same_as_prev = n_pairs = 0
        for run in IMAGERY_RUNS:
            (p,) = fetch_run_paths(s, [run])
            ann = mne.io.read_raw_edf(p, preload=False, verbose="error").annotations
            labs = [0 if a == "T1" else 1 for a in ann.description if a in ("T1", "T2")]
            for a, du in zip(ann.description, ann.duration):
                if a in ("T1", "T2"):
                    durs[0 if a == "T1" else 1].append(du)
            same_as_prev += sum(a == b for a, b in zip(labs[:-1], labs[1:]))
            n_pairs += len(labs) - 1
        rows.append({"subject": s,
                     "median_cue_s_left": np.median(durs[0]), "median_cue_s_right": np.median(durs[1]),
                     "p_same_class_as_previous": same_as_prev / n_pairs, "n_pairs": n_pairs})
    return pd.DataFrame(rows)


# Single-channel decoders: where on the scalp does the class information sit?
# Fewer CV repeats than the route decoders (3 vs 10) because there are 64
# channels x 3 feature types; this map is descriptive, no threshold is applied.
TOPO_FEATURES = {
    "low_freq_amplitude": dict(ref="raw", kind="erp", lowpass=4.0, window=(0.0, 3.975), bin_s=0.5,
                               baseline=(-0.5, 0.0)),
    "mu_beta_power": dict(ref="csd", kind="power", bands=[(8, 13), (13, 30)], separate_bands=True,
                          window=(0.5, 3.975), baseline=(-1.5, -0.5)),
    "high_freq_power": dict(ref="csd", kind="power", bands=[(30, 57), (63, 78)],
                            window=(0.5, 3.975), baseline=(-1.5, -0.5)),
}
TOPO_REPEATS = 3


def channel_topography(d: dict, Xref: dict, seed: int) -> dict:
    out = {}
    n_ch = len(d["ch_names"])
    for name, spec in TOPO_FEATURES.items():
        # Filter all channels once; features are laid out block-wise
        # (bin or band) x channel, so channel c is every n_ch-th column.
        F_all = route_features(d, dict(spec, channels="all"), Xref)
        accs = []
        for c in range(n_ch):
            F = F_all[:, c::n_ch]
            clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
            cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=TOPO_REPEATS, random_state=seed)
            acc = [ (clf.fit(F[tr], d["y"][tr]).predict(F[te]) == d["y"][te]).mean() for tr, te in cv.split(F, d["y"])]
            accs.append(np.mean(acc))
        out[name] = np.array(accs)
    return out


REF_PAIRS = [("F7", "F8"), ("FT7", "FT8"), ("T7", "T8"), ("AF7", "AF8"), ("Fp1", "Fp2")]


def reference_asymmetry(subjects) -> pd.DataFrame:
    """Where is the undocumented reference? The eyes are a left-right
    symmetric source, so the left/right class difference of the slow ocular
    signal should be equally large at mirror-image sites. Under the raw
    reference a lateral reference electrode that picks up EOG would cancel it
    on its own side. Ratio = |left-site diff| / |right-site diff| at
    0.25-0.6 s, median over subjects; CAR is the control."""
    res = {"raw": [], "car": []}
    for s in subjects:
        d = load_subject(s); ch = list(d["ch_names"]); t = d["times"]
        for ref in res:
            X = d["X"] if ref == "raw" else d["X"] - d["X"].mean(1, keepdims=True)
            x = _filt(X, None, 4.0)
            x = x - x[..., (t >= -0.5) & (t < 0)].mean(-1, keepdims=True)
            w = (t >= 0.25) & (t <= 0.6)
            diff = np.abs(x[d["y"] == 0][..., w].mean((0, 2)) - x[d["y"] == 1][..., w].mean((0, 2)))
            res[ref].append([diff[ch.index(a)] / diff[ch.index(b)] for a, b in REF_PAIRS])
    return pd.DataFrame({ref: np.median(np.array(v), axis=0) for ref, v in res.items()},
                        index=[f"{a}/{b}" for a, b in REF_PAIRS]).rename_axis("pair")


def run_a3(out_dir: Path, seed: int) -> pd.DataFrame:
    fig_dir = out_dir / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    subjects = kept_subjects(out_dir / "audit.csv")
    rng = np.random.default_rng(seed + 3000)
    rows, erp_store, topo = [], {"ocular": [], "posterior": []}, []
    for s in subjects:
        d = load_subject(s)
        y = d["y"]
        Xref = {"raw": d["X"], "csd": apply_reference(d["X"], d["ch_names"], "csd")}
        row = {"subject": s, "n_trials": len(y), "chance_threshold": chance_threshold(len(y))}
        y_shuf = rng.permutation(y)
        for name, route in ROUTES.items():
            F = route_features(d, route, Xref)
            row[f"acc_{name}"] = cv_accuracy(F, y, seed + s)
            row[f"shuffled_{name}"] = cv_accuracy(F, y_shuf, seed + s)
        F = order_features(d)
        row["acc_R5_order"] = cv_accuracy(F, y, seed + s)
        row["shuffled_R5_order"] = cv_accuracy(F, y_shuf, seed + s)
        rows.append(row)
        _store_erps(d, Xref, erp_store)
        topo.append(channel_topography(d, Xref, seed + s))
        print(f"A3 S{s:03d} (n={len(y)}, thr={row['chance_threshold']:.3f}): " +
              " ".join(f"{k[4:]}={v:.2f}" for k, v in row.items() if k.startswith("acc_")))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "a3_confound_decoders.csv", index=False, float_format="%.4f")

    timing = cue_timing_check(subjects)
    timing.to_csv(out_dir / "a3_cue_timing_and_sequence.csv", index=False, float_format="%.4f")

    asym = reference_asymmetry(subjects)
    asym.to_csv(out_dir / "a3_reference_asymmetry.csv", float_format="%.3f")
    print("ocular class-difference amplitude, left/right sites (median ratio):\n" + asym.round(2).to_string())

    summary = summarise(df, timing)
    summary.to_csv(out_dir / "a3_summary.csv", index=False, float_format="%.4g")
    print(summary.to_string(index=False))
    topo_mean = {k: np.mean([t[k] for t in topo], axis=0) for k in TOPO_FEATURES}
    pd.DataFrame(topo_mean, index=d["ch_names"]).rename_axis("channel").to_csv(
        out_dir / "a3_channel_topography.csv", float_format="%.4f")
    _figures(df, erp_store, d["times"], fig_dir, out_dir)
    _topo_figure(topo_mean, d["ch_names"], fig_dir)
    return df


def summarise(df: pd.DataFrame, timing: pd.DataFrame) -> pd.DataFrame:
    rows = []
    names = [c[4:] for c in df.columns if c.startswith("acc_")]
    for name in names:
        acc = df[f"acc_{name}"]; shuf = df[f"shuffled_{name}"]
        above = int((acc >= df.chance_threshold).sum())
        rows.append({
            "route": name,
            "mean_acc": acc.mean(), "sd_acc": acc.std(), "min_acc": acc.min(), "max_acc": acc.max(),
            "n_subjects_above_threshold": above,
            # How surprising is that count if no subject had the effect (5% each)?
            "p_count_vs_5pct": stats.binom.sf(above - 1, len(df), ALPHA),
            "p_wilcoxon_mean_gt_50": stats.wilcoxon(acc - 0.5, alternative="greater").pvalue,
            "shuffled_mean_acc": shuf.mean(),
            "shuffled_n_above_threshold": int((shuf >= df.chance_threshold).sum()),
        })
    out = pd.DataFrame(rows)
    # R6 and the sequence statistics as extra rows, so all routes sit in one table.
    diff = (timing.median_cue_s_left - timing.median_cue_s_right).abs().max()
    rows = [{"route": "R6_cue_timing", "note": f"max |median left - right cue duration| = {diff:.3f} s"},
            {"route": "R5_sequence",
             "note": f"P(class = previous class): mean {timing.p_same_class_as_previous.mean():.3f}, "
                     f"range {timing.p_same_class_as_previous.min():.2f}-{timing.p_same_class_as_previous.max():.2f}; "
                     f"binomial p (vs 0.5, pooled) = "
                     f"{stats.binomtest(int(round((timing.p_same_class_as_previous * timing.n_pairs).sum())), int(timing.n_pairs.sum()), 0.5).pvalue:.3g}"}]
    return pd.concat([out, pd.DataFrame(rows)], ignore_index=True)


def _store_erps(d, Xref, store):
    """Class-difference time courses for the mechanism figure."""
    ch = list(d["ch_names"]); y = d["y"]
    raw = _filt(Xref["raw"], None, 4.0)
    heog = raw[:, ch.index("F7")] - raw[:, ch.index("F8")]           # horizontal-EOG proxy
    heog = heog - heog[:, (d["times"] >= -0.5) & (d["times"] < 0)].mean(1, keepdims=True)
    store["ocular"].append(heog[y == 0].mean(0) - heog[y == 1].mean(0))
    csd = _filt(Xref["csd"], None, 20.0)
    occ = csd[:, ch.index("PO8")] - csd[:, ch.index("PO7")]          # right minus left posterior
    occ = occ - occ[:, (d["times"] >= -0.2) & (d["times"] < 0)].mean(1, keepdims=True)
    v = occ[y == 0].mean(0) - occ[y == 1].mean(0)
    store["posterior"].append(v / np.abs(v).max())                     # CSD units differ by subject


def _figures(df, erp_store, times, fig_dir, out_dir):
    _route_figure(df, fig_dir, out_dir)
    _mechanism_figure(erp_store, times, fig_dir)


def _route_figure(df, fig_dir, out_dir):
    style()
    names = [c[4:] for c in df.columns if c.startswith("acc_")]
    order = ["motor_sensorimotor", "R1_ocular", "R2_visual_evoked", "R3_posterior_alpha",
             "R4_emg", "R5_precue", "R5_precue_ocular", "R5_order"]
    order = [n for n in order if n in names]
    a2 = pd.read_csv(out_dir / "a2_lateralisation.csv").set_index("subject")
    lab = df.subject.map(a2["label"])
    fig, ax = plt.subplots(figsize=(10.0, 3.8))
    jitter = np.random.default_rng(0).uniform(-0.18, 0.18, len(df))
    for i, n in enumerate(order):
        for l, col, mk in (("present", C["blue"], "o"), ("absent", C["grey"], "s")):
            m = (lab == l).values
            ax.scatter(i + jitter[m], df[f"acc_{n}"].values[m], s=16, color=col, marker=mk,
                       edgecolor="white", linewidth=0.6, zorder=3,
                       label=f"A2: lateralised ERD {l}" if i == 0 else None)
        ax.hlines(df[f"acc_{n}"].mean(), i - 0.3, i + 0.3, color=C["ink"], lw=2, zorder=4)
    thr = df.chance_threshold
    # Thresholds differ by < 0.002 between subjects (n = 42 or 45): one line.
    ax.axhline(thr.max(), color=C["orange"], lw=1.2, ls="--", zorder=2)
    ax.axhline(0.5, color=C["ink2"], lw=0.8, ls=":")
    ax.text(len(order) - 0.45, thr.max(), f" p<0.05 chance\n threshold ({thr.max():.3f})",
            ha="left", va="center", fontsize=8, color=C["orange"])
    ax.text(len(order) - 0.45, 0.5, " 50%", ha="left", va="center", fontsize=8, color=C["ink2"])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([n.replace("_", "\n", 1) for n in order], fontsize=8)
    ax.set_ylabel("within-subject CV accuracy")
    ax.set_title("Restricted decoders per route (bar = mean of 37 subjects)", loc="left", pad=8)
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=8, borderaxespad=0.2)
    fig.savefig(fig_dir / "a3_route_accuracies.png")
    plt.close(fig)


def _mechanism_figure(erp_store, times, fig_dir):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.0))
    for ax, key, title, ylab in ((axes[0], "ocular", "Ocular: F7 − F8, left − right trials (< 4 Hz)", "µV"),
                                 (axes[1], "posterior", "Posterior: PO8 − PO7 (CSD), left − right (< 20 Hz)",
                                  "normalised")):
        a = np.array(erp_store[key])
        m, se = a.mean(0), a.std(0) / np.sqrt(len(a))
        ax.fill_between(times, m - 1.96 * se, m + 1.96 * se, color=C["blue"], alpha=0.25, lw=0)
        ax.plot(times, m, color=C["blue"], lw=1.5)
        ax.axhline(0, color=C["ink2"], lw=0.8); ax.axvline(0, color=C["ink"], lw=0.8)
        ax.set_title(title); ax.set_xlabel("time from cue (s)"); ax.set_ylabel(ylab)
        if key == "posterior":
            ax.set_xlim(-0.3, 1.0)
    fig.suptitle("Grand-average class difference (mean ± 95% CI over subjects)", y=1.02)
    fig.savefig(fig_dir / "a3_mechanisms.png")
    plt.close(fig)


def _topo_figure(topo_mean: dict, ch_names, fig_dir: Path):
    info = mne.create_info(list(ch_names), SFREQ, "eeg")
    info.set_montage("standard_1005")
    titles = {"low_freq_amplitude": "< 4 Hz amplitude (raw ref.)",
              "mu_beta_power": "μ/β power change (CSD)",
              "high_freq_power": "30–78 Hz power change (CSD)"}
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0))
    for ax, (k, v) in zip(axes, topo_mean.items()):
        im, _ = mne.viz.plot_topomap(v, info, axes=ax, show=False, cmap="Blues", vlim=(0.5, 0.65),
                                     contours=0, sensors=True)
        ax.set_title(titles[k])
    cb = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02)
    cb.set_label("single-channel accuracy (mean of 37 subjects)")
    fig.suptitle("Where the left/right information sits on the scalp", y=1.02)
    fig.savefig(fig_dir / "a3_channel_topography.png")
    plt.close(fig)


def per_subject_table(out_dir: Path) -> pd.DataFrame:
    """The brief's per-subject table: A2 lateralisation + A3 decoders + thresholds."""
    a2 = pd.read_csv(out_dir / "a2_lateralisation.csv")
    a3 = pd.read_csv(out_dir / "a3_confound_decoders.csv")
    cols2 = ["subject", "n_left", "n_right", "LI_mu_db", "p_mu", "LI_beta_db", "p_beta", "label"]
    acc_cols = [c for c in a3.columns if c.startswith("acc_")]
    t = a2[cols2].merge(a3[["subject", "n_trials", "chance_threshold"] + acc_cols], on="subject")
    for c in acc_cols:
        t[c.replace("acc_", "above_thr_")] = t[c] >= t["chance_threshold"]
    t = t.rename(columns={"label": "lateralised_ERD"})
    t.to_csv(out_dir / "per_subject_table.csv", index=False, float_format="%.4f")
    return t
