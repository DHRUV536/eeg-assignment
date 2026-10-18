"""A2: per-subject mu/beta ERD lateralisation at C3/C4, with a single-subject test.

Outputs (outputs/task1/):
  a2_lateralisation.csv     one row per subject: LI mu/beta, p, FDR q, label
  a2_reference_check.csv    how LI and labels change with the spatial reference
  a2_tf_S0xx.png            time-frequency maps for 2 'present' and 2 'absent' subjects
  a2_summary.png            LI mu vs beta for all subjects + grand-average ERD
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.time_frequency import tfr_array_morlet
from scipy.stats import false_discovery_control

from task1.data import kept_subjects, load_epochs

DISPLAY = (-1.5, 3.75)  # plots stop before the wavelet edge artefact at the epoch end

# ---- analysis parameters (all fixed before looking at any result) --------------------
LEFT_CH = ["C3", "FC3", "C1", "C5", "CP3"]   # C3 and its 4 nearest 10-10 neighbours
RIGHT_CH = ["C4", "FC4", "C2", "C6", "CP4"]  # mirror set over the right hemisphere
FREQS = np.arange(4.0, 36.0, 1.0)
# freq/2 cycles: ~0.5 s wavelets in mu, shorter in beta - enough frequency resolution to
# separate mu from beta while keeping temporal smearing below the 0.5 s guard intervals.
N_CYCLES = FREQS / 2.0
DECIM = 4                                    # 160 Hz -> 40 Hz TF time grid; ample for ERD
BANDS = {"mu": (8.0, 13.0), "beta": (13.0, 30.0)}
# Baseline inside the T0 rest that precedes every cue (4.1 s long). It ends 0.5 s before
# the cue so neither anticipation nor wavelet smearing of the cue response enters it, and
# starts ~2.6 s after the previous imagery ended, after most of the post-imagery beta
# rebound.
BASELINE = (-1.5, -0.5)
# Imagery window: skip the first 0.5 s (visual evoked response to the cue, and the
# wavelets' own smearing) and stop 0.5 s before the epoch edge at 4.0 s.
ACTIVE = (0.5, 3.5)
N_PERM = 5000
ALPHA = 0.05
SEED = 42
REFERENCES = ["as_recorded", "average", "csd"]  # csd is the primary analysis
PRIMARY_REF = "csd"


def spatial_reference(epochs: mne.Epochs, ref: str) -> mne.Epochs:
    ep = epochs.copy()
    if ref == "average":
        ep.set_eeg_reference("average", projection=False, verbose="error")
    elif ref == "csd":
        # Surface Laplacian: second spatial derivative of the potential, so any signal
        # common to all electrodes (including whatever the unknown reference picks up)
        # is removed, and each channel reflects mostly the cortex beneath it.
        ep = mne.preprocessing.compute_current_source_density(ep, verbose="error")
    return ep


def tf_power(epochs: mne.Epochs, picks: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Single-trial Morlet power, (trial, channel, freq, time), on a 40 Hz time grid."""
    x = epochs.get_data(picks=picks)
    p = tfr_array_morlet(x, epochs.info["sfreq"], FREQS, n_cycles=N_CYCLES,
                         output="power", decim=DECIM, verbose="error")
    return p, epochs.times[::DECIM]


def _mask(times, win):
    return (times >= win[0]) & (times <= win[1])


def band_erd(p: np.ndarray, times: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    """Per-trial ERD in dB: 10*log10(band power in the imagery window / band power in the
    same trial's pre-cue baseline), shape (trial, channel).

    Power is averaged over time and frequency *before* the log. Taking the log of each
    time-frequency pixel first and averaging afterwards biases every value downward
    (mean of log < log of mean for a noisy positive quantity), by ~2 dB here.
    Per-trial baseline: power drifts slowly within a 2-min run (alertness, impedance),
    and a trial's own rest is its closest reference. A baseline shared by all trials
    would cancel exactly in the left-minus-right LI, so the choice matters only through
    its noise.
    """
    f = (FREQS >= band[0]) & (FREQS <= band[1])
    act = p[:, :, f][..., _mask(times, ACTIVE)].mean(axis=(-1, -2))
    base = p[:, :, f][..., _mask(times, BASELINE)].mean(axis=(-1, -2))
    return 10 * np.log10(act / base)


def average_tf_db(p: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Trial-averaged ERD map (channel, freq, time): average power first, then dB
    relative to the averaged baseline of the same trials (classic ERD/ERS map)."""
    m = p.mean(axis=0)
    return 10 * np.log10(m / m[..., _mask(times, BASELINE)].mean(axis=-1, keepdims=True))


def lateralisation(d: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    """LI = mean(d | left) - mean(d | right), d = ERD(C3) - ERD(C4) in dB.

    Left-hand imagery should desynchronise the right hemisphere (C4 power drops), so d
    is larger on left trials: LI > 0 is the physiologically expected direction.
    Test: one-sided label permutation within the subject (exchangeable under H0: the
    label carries no information about d). Also returns LI / pooled SD (effect size).
    """
    def stat(lab):
        return d[lab == 0].mean() - d[lab == 1].mean()

    obs = stat(y)
    perm = np.array([stat(rng.permutation(y)) for _ in range(N_PERM)])
    p = (1 + np.sum(perm >= obs)) / (1 + N_PERM)
    pooled = np.sqrt((d[y == 0].var(ddof=1) + d[y == 1].var(ddof=1)) / 2)
    return float(obs), float(p), float(obs / pooled)


def analyse_subject(subject: int) -> tuple[list[dict], dict]:
    epochs = load_epochs(subject)
    y = (epochs.events[:, 2] == 2).astype(int)                  # 0 = left, 1 = right
    rows, tf_keep = [], {}
    for ref in REFERENCES:
        ep = spatial_reference(epochs, ref)
        pw, times = tf_power(ep, LEFT_CH + RIGHT_CH)
        row = dict(subject=subject, reference=ref, n_left=int((y == 0).sum()),
                   n_right=int((y == 1).sum()))
        for band, rng_ in BANDS.items():
            bw = band_erd(pw, times, rng_)
            d = bw[:, 0] - bw[:, len(LEFT_CH)]                       # C3 - C4
            # Seed per subject and band so each test is reproducible on its own.
            rng = np.random.default_rng([SEED, subject, list(BANDS).index(band)])
            li, p, es = lateralisation(d, y, rng)
            row.update({f"li_{band}_db": li, f"p_{band}": p, f"es_{band}": es,
                        # ERD in each hemisphere averaged over both classes, for context
                        f"erd_c3_{band}_db": float(bw[:, 0].mean()),
                        f"erd_c4_{band}_db": float(bw[:, len(LEFT_CH)].mean())})
            # Neighbourhood version (mean of the 5-channel sets): a robustness check
            # that the C3/C4 result is not one electrode's noise.
            dn = bw[:, :len(LEFT_CH)].mean(1) - bw[:, len(LEFT_CH):].mean(1)
            li_n, p_n, _ = lateralisation(dn, y, np.random.default_rng([SEED, subject, 10 + list(BANDS).index(band)]))
            row.update({f"li_{band}_neigh_db": li_n, f"p_{band}_neigh": p_n})
        rows.append(row)
        if ref == PRIMARY_REF:
            tf_keep = dict(times=times, left=average_tf_db(pw[y == 0], times),
                           right=average_tf_db(pw[y == 1], times))
    return rows, tf_keep


def add_fdr_and_labels(df: pd.DataFrame) -> pd.DataFrame:
    """BH-FDR across subjects, per band, separately for each reference.

    Label rule, fixed in advance: 'present' if the *mu* LI survives FDR (q < 0.05) in the
    expected direction (one-sided test, so direction is built in). The assignment defines
    the label on mu ERD; beta is kept as a supplementary column and does not change the
    label. FDR rather than raw p: with 37 subjects, ~2 would pass p < .05 by chance."""
    out = []
    for _, g in df.groupby("reference"):
        g = g.copy()
        g["q_mu"] = false_discovery_control(g.p_mu.values, method="bh")
        g["q_beta"] = false_discovery_control(g.p_beta.values, method="bh")
        g["label"] = np.where(g.q_mu < ALPHA, "present", "absent")
        out.append(g)
    return pd.concat(out)


def pick_examples(prim: pd.DataFrame) -> dict[str, list[int]]:
    """Two per label, chosen by rule rather than by eye: for 'present' the strongest and
    the weakest significant subject; for 'absent' the one closest to significance and
    the clearest null. Shows the range of each group, not only its best case."""
    # Ties in q are common after BH (step-up), so break them by effect size.
    prim = prim.assign(qmin=prim.q_mu, es=-prim.es_mu).sort_values(["qmin", "es"])
    pres, abs_ = prim[prim.label == "present"], prim[prim.label == "absent"]
    ex = {}
    if len(pres):
        ex["present"] = sorted({int(pres.subject.iloc[0]), int(pres.subject.iloc[-1])})
    if len(abs_):
        ex["absent"] = sorted({int(abs_.subject.iloc[0]), int(abs_.subject.iloc[-1])})
    return ex


def plot_subject(subject: int, tf: dict, row: pd.Series, path: Path) -> None:
    t = tf["times"]
    m = _mask(t, DISPLAY)
    i3, i4 = 0, len(LEFT_CH)
    panels = [("left fist", tf["left"]), ("right fist", tf["right"]),
              ("left - right", tf["left"] - tf["right"])]
    fig, ax = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True, constrained_layout=True)
    for r, (ci, cname) in enumerate([(i3, "C3"), (i4, "C4")]):
        for c, (title, arr) in enumerate(panels):
            lim = 6.0 if c < 2 else 4.0
            im = ax[r, c].pcolormesh(t[m], FREQS, arr[ci][:, m], cmap="RdBu_r",
                                     vmin=-lim, vmax=lim, shading="auto")
            for lo, hi in BANDS.values():
                ax[r, c].axhline(lo, color="k", lw=0.4, ls=":")
            ax[r, c].axhline(30, color="k", lw=0.4, ls=":")
            ax[r, c].axvline(0, color="k", lw=0.8)
            ax[r, c].axvspan(*BASELINE, color="grey", alpha=0.12)
            ax[r, c].add_patch(plt.Rectangle((ACTIVE[0], BANDS["mu"][0]), ACTIVE[1] - ACTIVE[0],
                                             BANDS["beta"][1] - BANDS["mu"][0], fill=False,
                                             ec="k", lw=0.6, ls="--"))
            ax[r, c].set_title(f"{cname} (CSD) - {title}", fontsize=9)
            if c == 0:
                ax[r, c].set_ylabel("Hz")
            if r == 1:
                ax[r, c].set_xlabel("time from cue (s)")
        fig.colorbar(im, ax=ax[r, :], shrink=0.8, label="dB vs baseline")
    fig.suptitle(f"S{subject:03d}  [{row.label}]  LI mu {row.li_mu_db:+.2f} dB (q={row.q_mu:.3f})   "
                 f"LI beta {row.li_beta_db:+.2f} dB (q={row.q_beta:.3f})   n={row.n_left}/{row.n_right}",
                 fontsize=10)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_summary(prim: pd.DataFrame, tfs: dict[int, dict], path: Path) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    col = prim.label.map({"present": "tab:blue", "absent": "tab:grey"})
    ax[0].scatter(prim.li_mu_db, prim.li_beta_db, c=col)
    for r in prim.itertuples():
        ax[0].annotate(str(r.subject), (r.li_mu_db, r.li_beta_db), fontsize=6, alpha=0.7)
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].axvline(0, color="k", lw=0.5)
    ax[0].set_xlabel("LI mu (dB)")
    ax[0].set_ylabel("LI beta (dB)")
    ax[0].set_title(f"Per-subject LI (CSD); blue = mu lateralised ERD present ({(prim.label == 'present').sum()}/{len(prim)})", fontsize=9)

    # Grand-average mu ERD time course, contralateral vs ipsilateral to the imagined hand.
    t = next(iter(tfs.values()))["times"]
    f = (FREQS >= BANDS["mu"][0]) & (FREQS <= BANDS["mu"][1])
    i3, i4 = 0, len(LEFT_CH)
    contra = np.mean([(tf["left"][i4][f].mean(0) + tf["right"][i3][f].mean(0)) / 2 for tf in tfs.values()], 0)
    ipsi = np.mean([(tf["left"][i3][f].mean(0) + tf["right"][i4][f].mean(0)) / 2 for tf in tfs.values()], 0)
    m = _mask(t, DISPLAY)
    ax[1].plot(t[m], contra[m], label="contralateral")
    ax[1].plot(t[m], ipsi[m], label="ipsilateral")
    ax[1].axvline(0, color="k", lw=0.8)
    ax[1].axvspan(*BASELINE, color="grey", alpha=0.12)
    ax[1].axvspan(*ACTIVE, color="tab:orange", alpha=0.06)
    ax[1].set_xlabel("time from cue (s)")
    ax[1].set_ylabel("mu power (dB)")
    ax[1].set_title(f"Grand average, all {len(tfs)} subjects (CSD)")
    ax[1].legend()

    fb = (FREQS >= BANDS["beta"][0]) & (FREQS <= BANDS["beta"][1])
    contra_b = np.mean([(tf["left"][i4][fb].mean(0) + tf["right"][i3][fb].mean(0)) / 2 for tf in tfs.values()], 0)
    ipsi_b = np.mean([(tf["left"][i3][fb].mean(0) + tf["right"][i4][fb].mean(0)) / 2 for tf in tfs.values()], 0)
    ax[2].plot(t[m], contra_b[m], label="contralateral")
    ax[2].plot(t[m], ipsi_b[m], label="ipsilateral")
    ax[2].axvline(0, color="k", lw=0.8)
    ax[2].axvspan(*BASELINE, color="grey", alpha=0.12)
    ax[2].axvspan(*ACTIVE, color="tab:orange", alpha=0.06)
    ax[2].set_xlabel("time from cue (s)")
    ax[2].set_ylabel("beta power (dB)")
    ax[2].set_title("Grand average beta (CSD)")
    ax[2].legend()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(out_dir: Path) -> pd.DataFrame:
    out_dir = Path(out_dir)
    rows, tfs = [], {}
    subjects = kept_subjects()
    for i, s in enumerate(subjects, 1):
        r, tf = analyse_subject(s)
        rows += r
        tfs[s] = tf
        print(f"[A2] {i:2d}/{len(subjects)} S{s:03d} done", flush=True)
    df = add_fdr_and_labels(pd.DataFrame(rows))

    prim = df[df.reference == PRIMARY_REF].drop(columns="reference").reset_index(drop=True)
    cols = ["subject", "n_left", "n_right", "li_mu_db", "es_mu", "p_mu", "q_mu",
            "li_beta_db", "es_beta", "p_beta", "q_beta", "label",
            "erd_c3_mu_db", "erd_c4_mu_db", "erd_c3_beta_db", "erd_c4_beta_db",
            "li_mu_neigh_db", "p_mu_neigh", "li_beta_neigh_db", "p_beta_neigh"]
    prim[cols].round(4).to_csv(out_dir / "a2_lateralisation.csv", index=False)

    # Reference sensitivity: LI and label under each reference, side by side.
    wide = df.pivot(index="subject", columns="reference", values=["li_mu_db", "li_beta_db", "label"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    num = [c for c in wide.columns if not c.startswith("label")]
    wide[num] = wide[num].astype(float).round(4)
    wide.reset_index().to_csv(out_dir / "a2_reference_check.csv", index=False)

    # Example subjects depend on the results; clear old ones so no stale figure survives.
    for old in out_dir.glob("a2_tf_S*.png"):
        old.unlink()
    for label, subs in pick_examples(prim).items():
        for s in subs:
            plot_subject(s, tfs[s], prim[prim.subject == s].iloc[0],
                         out_dir / f"a2_tf_S{s:03d}_{label}.png")
    plot_summary(prim, tfs, out_dir / "a2_summary.png")

    for ref in REFERENCES:
        g = df[df.reference == ref]
        print(f"[A2] {ref:12s} present {int((g.label == 'present').sum()):2d}/{len(g)}  "
              f"(mu p<.05 uncorrected: {int((g.p_mu < ALPHA).sum())}; beta q<.05, supplementary: {int((g.q_beta < ALPHA).sum())}); "
              f"median LI mu {g.li_mu_db.median():+.2f} dB, beta {g.li_beta_db.median():+.2f} dB")
    ch = (wide[f"label_{PRIMARY_REF}"] != wide["label_as_recorded"]).sum()
    print(f"[A2] labels that change between as-recorded and CSD reference: {ch}")
    print(f"[A2] examples: {pick_examples(prim)}")
    return prim


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[1] / "outputs" / "task1")
