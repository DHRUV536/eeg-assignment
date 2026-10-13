"""
A2 - Physiological ground truth: cue-locked mu/beta ERD at C3/C4 and neighbours.

Reference: the recording reference is undocumented and ~25-60% of every
channel's variance is common to all channels (DECISIONS D9). Power at C3 and
C4 is therefore measured against an unknown, possibly active, possibly
lateral site. We use the spherical-spline surface Laplacian (current source
density, CSD), which is reference-free and local, and repeat the whole
analysis under common average reference (CAR) and the raw reference to show
how much the labels depend on that choice.

Lateralisation index (per subject, per band):
    e_ch(trial) = band power in 0.5-3.975 s, in dB relative to the
                  subject's mean baseline power (-1.5 to -0.5 s, all trials)
    h(trial)    = e_C4 - e_C3       (positive = relatively more ERD on the left)
    LI          = mean_right(h) - mean_left(h)   [dB]
Motor imagery predicts ERD contralateral to the imagined hand, so LI > 0.
Test: one-sided label-permutation test of LI > 0 within the subject.
"""
from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne.time_frequency import tfr_array_morlet

from task1.data import SFREQ, TMIN, kept_subjects, load_subject
from task1.plotting import C, DIVERGING, plt, style

# C3, C4 and their neighbours in the 10-10 grid (the brief's "and neighbours").
TF_CHANNELS = ["FC3", "C5", "C3", "C1", "CP3", "Cz", "FC4", "C6", "C4", "C2", "CP4"]
FREQS = np.arange(4.0, 41.0, 1.0)
# freq/2 cycles = a 0.5 s wavelet at every frequency: fine enough in time to
# keep the 0.5 s post-cue exclusion meaningful, ~2 Hz resolution in mu.
N_CYCLES = FREQS / 2.0
DECIM = 4                                   # 40 Hz time grid is plenty for 4-40 Hz power

# Baseline inside the 4.1 s rest period that precedes every cue: it starts
# 2.6 s after the previous imagery ended (post-imagery beta rebound is over
# within ~1-2 s) and ends 0.5 s before the cue, so the 0.5 s wavelets do not
# smear the cue response into it, and anticipation of a cue whose timing is
# fixed (4.1 s rhythm) has less room to appear.
BASELINE = (-1.5, -0.5)
# ERD window starts 0.5 s after the cue, skipping the visual evoked response
# to the target appearing (an A3 route) and the wavelet's onset smearing.
ERD_WIN = (0.5, 3.975)
BANDS = {"mu": (8.0, 13.0), "beta": (13.0, 30.0)}
N_PERM = 10_000
# Two bands are tested per subject, so each band is tested at alpha/2
# (Bonferroni) and the per-subject label keeps a family-wise alpha of 0.05.
ALPHA = 0.05
ALPHA_BAND = ALPHA / len(BANDS)
REFERENCES = ("csd", "car", "raw")


def apply_reference(X: np.ndarray, ch_names, kind: str) -> np.ndarray:
    """Re-reference trials (n, ch, t) in microvolts."""
    if kind == "raw":
        return X
    if kind == "car":
        return X - X.mean(axis=1, keepdims=True)
    if kind == "csd":
        info = mne.create_info(list(ch_names), SFREQ, "eeg")
        ep = mne.EpochsArray(X * 1e-6, info, tmin=TMIN, verbose="error")
        ep.set_montage("standard_1005", verbose="error")
        # MNE defaults (spline order 4, lambda 1e-5) as in Perrin et al. 1989.
        ep = mne.preprocessing.compute_current_source_density(ep, verbose="error")
        return ep.get_data()                # V/m^2; units cancel in the dB ratio
    raise ValueError(kind)


def tf_power(Xch: np.ndarray, times: np.ndarray):
    """Single-trial Morlet power (n, ch, f, t) and the subject's baseline (ch, f)."""
    power = tfr_array_morlet(Xch, SFREQ, FREQS, n_cycles=N_CYCLES, output="power",
                             decim=DECIM, verbose="error")
    t = times[::DECIM]
    base_mask = (t >= BASELINE[0]) & (t <= BASELINE[1])
    # Baseline pooled over all trials of both classes: a per-class baseline
    # could itself create a left/right difference.
    base = power[..., base_mask].mean(axis=-1).mean(axis=0)
    return power, base, t


def band_erd(power: np.ndarray, base: np.ndarray, t: np.ndarray, band) -> np.ndarray:
    """Per-trial band ERD in dB: power is averaged over the band and window FIRST,
    then converted to dB. Averaging single-trial dB instead is biased by about
    -2.5 dB (E[log P] < log E[P] for skewed power), which showed up as a fake
    'ERD' in the baseline period of the first version of the figures."""
    fmask = (FREQS >= band[0]) & (FREQS < band[1])
    tmask = (t >= ERD_WIN[0]) & (t <= ERD_WIN[1])
    trial_bp = power[:, :, fmask][..., tmask].mean(axis=(2, 3))     # (n, ch)
    base_bp = base[:, fmask].mean(axis=1)                           # (ch,)
    return 10 * np.log10(trial_bp / base_bp[None, :])


def lateralisation_test(h: np.ndarray, y: np.ndarray, rng: np.random.Generator):
    """LI = mean(h | right) - mean(h | left); one-sided permutation p for LI > 0."""
    def li(labels):
        r = labels == 1
        return h[r].mean() - h[~r].mean()

    obs = li(y)
    # Vectorised permutations: each row is a shuffled label vector.
    perms = np.array([rng.permutation(y) for _ in range(N_PERM)])
    r = perms == 1
    null = (r * h).sum(1) / r.sum(1) - ((~r) * h).sum(1) / (~r).sum(1)
    p = (1 + np.sum(null >= obs)) / (N_PERM + 1)
    return obs, p


def analyse_subject(d: dict, reference: str, seed: int) -> tuple[dict, dict]:
    ch = list(d["ch_names"])
    Xr = apply_reference(d["X"], ch, reference)
    idx = [ch.index(c) for c in TF_CHANNELS]
    power, base, t = tf_power(Xr[:, idx], d["times"])
    y = d["y"]
    i3, i4 = TF_CHANNELS.index("C3"), TF_CHANNELS.index("C4")
    rng = np.random.default_rng(seed)
    row = {}
    present = False
    for band_name, band in BANDS.items():
        e = band_erd(power, base, t, band)
        h = e[:, i4] - e[:, i3]
        li, p = lateralisation_test(h, y, rng)
        # ERD contralateral to the imagined hand: C4 on left trials, C3 on right.
        contra = np.concatenate([e[y == 0, i4], e[y == 1, i3]]).mean()
        row[f"LI_{band_name}_db"] = li
        row[f"p_{band_name}"] = p
        row[f"contra_erd_{band_name}_db"] = contra
        # "Lateralised ERD": the asymmetry is significant AND it is a power
        # decrease on the contralateral side, not only an ipsilateral increase.
        present |= (p < ALPHA_BAND) and (contra < 0)
    row["label"] = "present" if present else "absent"
    # Class-average maps for the figures: (class, C3/C4, f, t)
    maps = np.stack([10 * np.log10(power[y == k][:, [i3, i4]].mean(0) / base[[i3, i4], :, None])
                     for k in (0, 1)])
    return row, {"maps": maps, "t": t}


def run_a2(out_dir: Path, seed: int) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"; fig_dir.mkdir(exist_ok=True)
    subjects = kept_subjects(out_dir / "audit.csv")
    rows, maps = [], {}
    for s in subjects:
        d = load_subject(s)
        row = {"subject": s, "n_left": int((d["y"] == 0).sum()), "n_right": int((d["y"] == 1).sum())}
        for ref in REFERENCES:
            # Same seed per subject for every reference: the permutation noise
            # is then identical and differences come from the reference only.
            r, m = analyse_subject(d, ref, seed + s)
            suffix = "" if ref == "csd" else f"_{ref}"
            row.update({k + suffix: v for k, v in r.items()})
            if ref == "csd":
                maps[s] = m
        rows.append(row)
        print(f"A2 S{s:03d}: mu LI {row['LI_mu_db']:+.2f} dB p={row['p_mu']:.4f} | "
              f"beta LI {row['LI_beta_db']:+.2f} dB p={row['p_beta']:.4f} -> {row['label']}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "a2_lateralisation.csv", index=False, float_format="%.5g")
    # Large intermediate (class-average TF maps): kept in cache/, not in git.
    cache = out_dir.parents[1] / "cache"; cache.mkdir(exist_ok=True)
    np.savez(cache / "a2_maps.npz", subjects=np.array(list(maps)),
             maps=np.stack([maps[s]["maps"] for s in maps]), t=maps[subjects[0]]["t"], freqs=FREQS)

    _figures(df, maps, fig_dir)
    agree = {ref: (df["label"] == df[f"label_{ref}"]).mean() for ref in ("car", "raw")}
    n_present = (df["label"] == "present").sum()
    print(f"A2: lateralised ERD present in {n_present}/{len(df)} subjects (CSD); "
          f"label agreement with CAR {agree['car']:.0%}, raw {agree['raw']:.0%}")
    return df


def _pick_examples(df: pd.DataFrame):
    """3 present (strongest evidence) and 3 absent (weakest evidence), chosen by rule."""
    df = df.assign(p_min=df[["p_mu", "p_beta"]].min(axis=1))
    present = df[df.label == "present"].sort_values("p_min").subject.tolist()[:3]
    absent = df[df.label == "absent"].sort_values("p_min", ascending=False).subject.tolist()[:3]
    return present, absent


def _figures(df: pd.DataFrame, maps: dict, fig_dir: Path):
    style()
    present, absent = _pick_examples(df)
    for name, subs in (("present", present), ("absent", absent)):
        fig, axes = plt.subplots(len(subs), 4, figsize=(9.5, 2.0 * len(subs)), sharex=True, sharey=True,
                                 squeeze=False)
        for i, s in enumerate(subs):
            m, t = maps[s]["maps"], maps[s]["t"]
            r = df.set_index("subject").loc[s]
            for j, (k, c, title) in enumerate([(0, 0, "C3 · left"), (1, 0, "C3 · right"),
                                               (0, 1, "C4 · left"), (1, 1, "C4 · right")]):
                ax = axes[i, j]
                im = ax.pcolormesh(t, FREQS, m[k, c], cmap=DIVERGING, vmin=-3, vmax=3, shading="auto")
                ax.axvline(0, color=C["ink"], lw=0.8)
                for lo, hi in BANDS.values():
                    ax.axhline(lo, color=C["ink2"], lw=0.4, ls=":")
                ax.grid(False)
                if i == 0:
                    ax.set_title(title)
                if j == 0:
                    ax.set_ylabel(f"S{s:03d}\nHz")
                if i == len(subs) - 1:
                    ax.set_xlabel("time from cue (s)")
            axes[i, 3].text(1.03, 0.5, f"μ LI {r.LI_mu_db:+.2f} dB, p={r.p_mu:.3f}\n"
                            f"β LI {r.LI_beta_db:+.2f} dB, p={r.p_beta:.3f}",
                            transform=axes[i, 3].transAxes, va="center", fontsize=8, color=C["ink2"])
        cb = fig.colorbar(im, ax=axes, location="bottom", shrink=0.4, pad=0.12)
        cb.set_label("power vs baseline (dB); blue = ERD")
        fig.suptitle(f"Lateralised ERD {name}: CSD, class-average time-frequency power", y=1.0)
        fig.savefig(fig_dir / f"a2_tf_{name}.png")
        plt.close(fig)

    # Group view: mu vs beta LI per subject, label shown by marker AND colour.
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    for lab, col, mk in (("present", C["blue"], "o"), ("absent", C["grey"], "s")):
        sub = df[df.label == lab]
        ax.scatter(sub.LI_mu_db, sub.LI_beta_db, s=36, color=col, marker=mk,
                   edgecolor="white", linewidth=1, label=f"{lab} (n={len(sub)})", zorder=3)
    ax.axhline(0, color=C["ink2"], lw=0.8); ax.axvline(0, color=C["ink2"], lw=0.8)
    ax.set_xlabel("μ (8–13 Hz) lateralisation index (dB)")
    ax.set_ylabel("β (13–30 Hz) lateralisation index (dB)")
    ax.set_title("Per-subject lateralisation (CSD); > 0 = contralateral ERD")
    ax.legend(loc="upper left")
    fig.savefig(fig_dir / "a2_li_scatter.png")
    plt.close(fig)

    # Reference sensitivity: the same index under CAR and raw reference.
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4), sharey=True)
    for ax, band in zip(axes, BANDS):
        x = np.arange(len(df))
        order = np.argsort(df[f"LI_{band}_db"].values)
        for ref, col, mk in (("csd", C["blue"], "o"), ("car", C["orange"], "^"), ("raw", C["grey"], "s")):
            key = f"LI_{band}_db" + ("" if ref == "csd" else f"_{ref}")
            ax.scatter(x, df[key].values[order], s=18, color=col, marker=mk, label=ref.upper(), zorder=3)
        ax.axhline(0, color=C["ink2"], lw=0.8)
        ax.set_title(f"{band} LI by reference")
        ax.set_xlabel("subjects, sorted by CSD LI")
    axes[0].set_ylabel("lateralisation index (dB)")
    axes[0].legend()
    fig.savefig(fig_dir / "a2_reference_effect.png")
    plt.close(fig)
