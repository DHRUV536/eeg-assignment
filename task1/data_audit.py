"""A1 data audit: one row per subject (audit.csv) plus one row per run (audit_runs.csv).

Every anomaly found here must have an explicit decision in SUBJECT_DECISIONS, or the
audit stops with an error. That is how "nothing is dropped or kept silently" is enforced.
"""
from __future__ import annotations

import os
from pathlib import Path

import mne
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Task definition (verified on the PhysioNet EEGMMIDB page,
# https://physionet.org/content/eegmmidb/1.0.0/ , "Experimental Protocol" section)
# --------------------------------------------------------------------------------------
SUBJECTS = list(range(70, 110))  # 70–109 inclusive, fixed by the assignment
# Runs 4, 8, 12 = task 2, *imagine* opening/closing left or right fist.
IMAGERY_RUNS = [4, 8, 12]
# In runs 4/8/12, T1 = left fist and T2 = right fist; T0 = rest (dataset page).
EVENT_MEANING = {"T0": "rest", "T1": "left_fist_imagery", "T2": "right_fist_imagery"}
EVENT_MAPPING_STR = "T0=rest(dropped); T1=left fist imagery -> 0; T2=right fist imagery -> 1"

# What a "normal" run looks like, so anything else is flagged (from the dataset page and
# confirmed on the majority of files: 160 Hz, 64 ch, ~2 min, 15 cues of ~4.1 s each).
EXPECTED_SFREQ = 160.0
EXPECTED_NCH = 64
EXPECTED_TRIALS_PER_RUN = 15
EXPECTED_TASK_DUR = 4.1
EXPECTED_RUN_DUR = 123.0
DUR_TOL = 0.15  # s; a 0.1 s rounding of annotation length is harmless for a 4 s window

# The window later analyses will cut around each cue. A trial is only usable if this
# whole window lies inside the recording, and the pre-cue part should fall in rest
# (needed for the A2 baseline, which must end before the cue). The window starts 0.5 s
# before the A2 baseline (-1.5 s) so wavelet edge effects fall outside it, and ends at
# 4.0 s because the last cue of every run is only ~4.1 s from the end of the file.
TRIAL_WINDOW = (-2.0, 4.0)

# Channel sanity: compare each channel's std to the run median, because absolute
# amplitudes differ a lot between subjects (impedance, skull) but within a run a
# healthy channel stays within a few-fold of its neighbours.
FLAT_RATIO = 0.2
NOISY_RATIO = 5.0

# --------------------------------------------------------------------------------------
# Decisions for anomalous subjects. Each anomaly flag must be covered here.
# action: "keep" | "keep_interp" | "keep_partial" | "exclude"
# These are the current choices; mirror each one in DECISIONS.md with the alternative
# and what you expect would change.
# --------------------------------------------------------------------------------------
BENIGN_FLAGS = {
    # Annotation durations of 4.0 or 4.15 s instead of 4.1 s: a rounding of the cue
    # length, smaller than one 250 ms analysis step, and our window ends at 4.0 s.
    "task_dur_rounding",
}
SUBJECT_DECISIONS: dict[int, tuple[str, str]] = {
    88: ("exclude", "Recorded at 128 Hz with a different cue timing (5.12 s cues, ~1.4 s rest), "
                    "so there is no rest period long enough for the pre-cue baseline and the "
                    "trial timing is not comparable to the 160 Hz protocol. Resampling fixes the "
                    "rate but not the timing."),
    92: ("exclude", "Same as S088: 128 Hz and non-standard cue/rest timing."),
    100: ("exclude", "128 Hz, non-standard timing (5.12 s cues and rests) and only 12 cues per "
                     "run (6 left / 6 right) -> 18 trials per class, too few for a stable "
                     "single-subject test."),
    # Single dead/noisy channels (std < 0.2x or > 5x the run median). One bad electrode
    # does not justify losing a subject; downstream stages read `bad_channels` and
    # interpolate them (spherical spline) from neighbours. Note which analyses they touch.
    79: ("keep_interp", "Fp1 near-flat in runs 8/12, P1 noisy in run 4; interpolate. Fp1 feeds "
                        "the A3 ocular decoder, so check S079 is not driving that result."),
    86: ("keep_interp", "Fp1 near-flat in runs 8/12; interpolate (A3 ocular-decoder caveat as S079)."),
    97: ("keep_interp", "AF7 near-flat in runs 4/12; interpolate (frontal, A3 ocular-decoder caveat)."),
    105: ("keep_interp", "FC5 and FC1 near-flat in all runs; interpolate. Both are C3 neighbours, "
                         "so the C3 Laplacian/CSD for this subject leans on interpolated values."),
    104: ("keep_partial", "Run 8 is truncated (~106 s instead of 123 s). Trials whose analysis "
                          "window is fully inside the recording are kept; the rest of the "
                          "subject's data is normal."),
}


# --------------------------------------------------------------------------------------
def data_dir() -> Path | None:
    """Local PhysioNet folder: $EEGBCI_DIR, else ../files next to this repository."""
    env = os.environ.get("EEGBCI_DIR")
    if env:
        return Path(env)
    local = Path(__file__).resolve().parents[2] / "files"
    return local if local.exists() else None


def run_path(subject: int, run: int) -> Path:
    """Use the local copy when present; otherwise download through MNE so that
    `python main.py` works on a fresh machine with no manual step."""
    root = data_dir()
    if root is not None:
        p = root / f"S{subject:03d}" / f"S{subject:03d}R{run:02d}.edf"
        if p.exists():
            return p
    return Path(mne.datasets.eegbci.load_data(subject, [run], update_path=False, verbose=False)[0])


def load_run(subject: int, run: int) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_edf(run_path(subject, run), preload=True, verbose="ERROR")
    # Channel names in the EDF carry trailing dots ("Fc5."); standardise so montage and
    # later channel picks (C3, C4, ...) work.
    mne.datasets.eegbci.standardize(raw)
    return raw


def audit_run(subject: int, run: int) -> dict:
    raw = load_run(subject, run)
    sf = float(raw.info["sfreq"])
    dur = raw.n_times / sf
    ann = raw.annotations
    desc = np.asarray(ann.description)
    onset, adur = np.asarray(ann.onset), np.asarray(ann.duration)

    unknown = sorted(set(desc) - set(EVENT_MEANING))
    task = np.isin(desc, ["T1", "T2"])
    rest = desc == "T0"

    # A trial is usable if the whole analysis window fits in the recording.
    t0, t1 = TRIAL_WINDOW
    fits = (onset + t0 >= 0) & (onset + t1 <= dur)
    usable_left = int(np.sum(fits & (desc == "T1")))
    usable_right = int(np.sum(fits & (desc == "T2")))

    # Rest immediately before each cue (the baseline has to come from it).
    pre_rest = []
    for i in np.flatnonzero(task):
        if i > 0 and desc[i - 1] == "T0":
            pre_rest.append(adur[i - 1])
    min_pre_rest = float(np.min(pre_rest)) if pre_rest else np.nan

    data = raw.get_data(picks="eeg")
    sd = data.std(axis=1)
    med = np.median(sd)
    flat = [raw.ch_names[i] for i in np.flatnonzero(sd < FLAT_RATIO * med)]
    noisy = [raw.ch_names[i] for i in np.flatnonzero(sd > NOISY_RATIO * med)]

    flags = []
    if sf != EXPECTED_SFREQ:
        flags.append(f"sfreq_{sf:g}Hz")
    if len(raw.ch_names) != EXPECTED_NCH:
        flags.append(f"n_channels_{len(raw.ch_names)}")
    if unknown:
        flags.append(f"unknown_events_{'/'.join(unknown)}")
    if task.sum() != EXPECTED_TRIALS_PER_RUN:
        flags.append(f"n_cues_{int(task.sum())}")
    if dur < EXPECTED_RUN_DUR - 2:
        flags.append(f"truncated_{dur:.0f}s")
    if (onset[-1] + adur[-1]) > dur + 1e-3:
        flags.append("events_past_end")
    task_d = np.unique(np.round(adur[task], 2))
    off = np.abs(task_d - EXPECTED_TASK_DUR)
    if np.any(off > DUR_TOL):
        flags.append(f"task_dur_{'/'.join(f'{d:g}' for d in task_d)}s")
    elif np.any(off > 1e-6):
        flags.append("task_dur_rounding")
    if np.isfinite(min_pre_rest) and min_pre_rest < -t0:
        flags.append(f"short_rest_{min_pre_rest:.2f}s")
    if usable_left + usable_right < task.sum():
        flags.append(f"trials_outside_window_{int(task.sum()) - usable_left - usable_right}")
    if flat:
        flags.append(f"flat_ch_{'/'.join(flat)}")
    if noisy:
        flags.append(f"noisy_ch_{'/'.join(noisy)}")

    return dict(
        subject=subject, run=run, sfreq=sf, n_channels=len(raw.ch_names),
        duration_s=round(dur, 2), n_left=int((desc == "T1").sum()),
        n_right=int((desc == "T2").sum()), n_rest=int(rest.sum()),
        usable_left=usable_left, usable_right=usable_right,
        task_durations_s="/".join(f"{d:g}" for d in task_d),
        min_pre_cue_rest_s=round(min_pre_rest, 2),
        median_ch_std_uV=round(med * 1e6, 1),
        anomalies=";".join(flags),
    )


def summarise_subject(runs: pd.DataFrame) -> dict:
    s = int(runs.subject.iloc[0])
    flags = sorted({f for a in runs.anomalies for f in a.split(";") if f})
    serious = [f for f in flags if f not in BENIGN_FLAGS]
    if s in SUBJECT_DECISIONS:
        action, reason = SUBJECT_DECISIONS[s]
    elif not serious:
        action = "keep"
        reason = "no anomalies" if not flags else "only benign annotation-length rounding"
    else:
        raise RuntimeError(
            f"S{s:03d} has anomalies {serious} but no entry in SUBJECT_DECISIONS. "
            "Decide keep/exclude explicitly (and record it in DECISIONS.md)."
        )
    sfreqs = sorted(runs.sfreq.unique())
    return dict(
        subject=s,
        runs_loaded="/".join(str(r) for r in runs.run),
        sfreq="/".join(f"{f:g}" for f in sfreqs),
        duration_s=round(runs.duration_s.sum(), 1),
        n_left=int(runs.n_left.sum()), n_right=int(runs.n_right.sum()),
        usable_left=int(runs.usable_left.sum()), usable_right=int(runs.usable_right.sum()),
        event_mapping=EVENT_MAPPING_STR,
        bad_channels="/".join(sorted({c for a in runs.anomalies for f in a.split(";")
                                      if f.startswith(("flat_ch_", "noisy_ch_"))
                                      for c in f.split("_", 2)[2].split("/")})),
        anomalies="; ".join(f"R{r.run:02d}:{r.anomalies}" for r in runs.itertuples() if r.anomalies),
        action=action,
        action_reason=reason,
    )


def bad_channels(audit_csv: Path) -> dict[int, list[str]]:
    """Per-subject channels to interpolate before any analysis."""
    df = pd.read_csv(audit_csv).fillna("")
    return {int(r.subject): r.bad_channels.split("/") for r in df.itertuples() if r.bad_channels}


def kept_subjects(audit_csv: Path) -> list[int]:
    """Subjects later stages should use (keep or keep_partial)."""
    df = pd.read_csv(audit_csv)
    return df.loc[df.action != "exclude", "subject"].astype(int).tolist()


def main(out_dir: Path) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = data_dir()
    print(f"[A1] reading EDFs from {src if src else 'MNE download cache'}")

    run_rows = []
    for s in SUBJECTS:
        for r in IMAGERY_RUNS:
            run_rows.append(audit_run(s, r))
    runs = pd.DataFrame(run_rows)
    runs.to_csv(out_dir / "audit_runs.csv", index=False)

    subj = pd.DataFrame([summarise_subject(g) for _, g in runs.groupby("subject")])
    subj.to_csv(out_dir / "audit.csv", index=False)

    # Decisions for subjects that turned out clean are stale; say so rather than hide it.
    for s in SUBJECT_DECISIONS:
        if not subj.loc[subj.subject == s, "anomalies"].iloc[0]:
            print(f"[A1] WARNING: S{s:03d} has a decision but no anomalies")

    kept = subj[subj.action != "exclude"]
    print(f"[A1] {len(subj)} subjects audited; kept {len(kept)}, "
          f"excluded {sorted(subj.loc[subj.action == 'exclude', 'subject'].tolist())}")
    print(f"[A1] usable trials in kept subjects: left={kept.usable_left.sum()} "
          f"right={kept.usable_right.sum()}")
    flagged = subj[subj.anomalies != ""]
    for row in flagged.itertuples():
        print(f"  S{row.subject:03d} [{row.action}] {row.anomalies}")
    print(f"[A1] wrote {out_dir / 'audit.csv'} and {out_dir / 'audit_runs.csv'}")
    return subj


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[1] / "outputs" / "task1")
