"""
A1 - Data audit for PhysioNet EEG Motor Movement/Imagery (EEGBCI).

Subjects 70-109, imagined left fist vs imagined right fist.
Writes:
  audit.csv       one row per subject (what the brief asks for)
  audit_runs.csv  one row per run (the evidence behind each subject row)

Nothing is dropped silently: every anomaly is listed, and the action taken
comes from ACTION_POLICY below, so the decision is visible in one place and
can be changed and re-run.
"""
from __future__ import annotations

import csv
import os
from functools import lru_cache
from pathlib import Path

import mne
import numpy as np
from mne.datasets import eegbci

# Local copy of the EDFs. Default: the `files` folder next to the repo
# (".../EEG Assignment/../files"). Override with the EEGBCI_DIR env var.
# Any nesting below it works (files/S070/S070R04.edf, files/eegmmidb/1.0.0/...),
# because files are found by their PhysioNet name, not by a fixed path.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("EEGBCI_DIR", REPO_ROOT.parent / "files"))

# --------------------------------------------------------------------------
# Dataset facts. Verify each against the PhysioNet eegmmidb page
# (https://physionet.org/content/eegmmidb/1.0.0/, "Experimental Protocol")
# and cite it in DECISIONS.md.
# --------------------------------------------------------------------------
SUBJECTS = list(range(70, 110))  # 70-109 inclusive, as the brief specifies

# Runs 4, 8, 12 are the imagined left/right fist task. Runs 3, 7, 11 are the
# same task executed, which the brief excludes.
IMAGERY_RUNS = [4, 8, 12]

# T1/T2 mean different things in different run types (in runs 5,6,9,10,13,14
# they are both fists / both feet), so this mapping is only valid because we
# load runs 4, 8, 12 exclusively.
EVENT_MAP = {"T0": "rest", "T1": "left", "T2": "right"}

NOMINAL_SFREQ = 160.0  # stated on the dataset page
NOMINAL_N_CH = 64      # stated on the dataset page

# --------------------------------------------------------------------------
# Anomaly thresholds. These are judgement calls; each one is a DECISIONS.md
# entry. Run-level checks compare against the median over all audited runs,
# because the protocol fixes run structure but not exact sample counts.
# --------------------------------------------------------------------------
RUN_DURATION_TOL_S = 5.0     # a run differing by >5 s from the median is not the standard ~2-min run
CUE_DURATION_TOL_S = 0.1     # cue timing is protocol-fixed, so even 0.1 s means a different protocol timing
FLAT_SD_V = 0.5e-6           # <0.5 uV SD over a whole run is not physiological EEG: disconnected electrode
HUGE_PTP_V = 1000e-6         # >1 mV after 1 Hz high-pass is far above EEG, likely a lost contact
IMBALANCE_FRAC = 0.2         # |L-R|/(L+R) above this would bias a classifier towards one class
MIN_TRIALS_PER_CLASS = 15    # below this, per-subject statistics in A2/A3 have no power

# What to do with a subject for each anomaly type. One of "keep", "resample",
# "drop". The subject's overall action is the most severe one triggered
# (drop > resample > keep). Every entry here is a decision to justify.
ACTION_POLICY = {
    "missing_run": "drop",          # fewer trials and an unbalanced design across sessions
    # Inspection showed every 128 Hz subject (88, 92, 100) also has different
    # trial timing (5.125 s cues; 1.375 s rest in 88/92), so resampling would
    # fix the rate but not make the trials comparable. See DECISIONS.md D3.
    "sfreq_mismatch": "drop",
    "protocol_timing": "drop",      # median cue length differs: a different trial structure
    "channel_mismatch": "drop",     # cross-subject models need an identical montage
    "too_few_trials": "drop",       # see MIN_TRIALS_PER_CLASS
    "run_duration": "keep",         # length alone does not change trial content; flagged only
    "cue_count": "keep",            # handled by trial counts; flagged only
    "truncated_trial": "keep",      # recording ends mid-trial; that one trial is lost at epoching
    "unknown_event_codes": "keep",  # unknown codes are ignored when epoching
    "events_past_end": "keep",      # those trials are dropped at epoching, counts reflect that
    # Almost always Fp/AF channels = eye movements and blinks, not broken
    # electrodes, so they are NOT interpolated (that would hide the ocular
    # signal A3 needs to test). Gross artifacts are rejected per trial instead.
    "high_amplitude_channels": "keep",
    "class_imbalance": "keep",      # chance level is computed from actual counts later
}
_SEVERITY = {"keep": 0, "resample": 1, "drop": 2}


@lru_cache(maxsize=1)
def _local_index() -> dict[str, Path]:
    """Map 'S070R04.edf' -> path for every EDF under DATA_DIR (scanned once)."""
    if not DATA_DIR.is_dir():
        return {}
    return {p.name.upper(): p for p in DATA_DIR.rglob("*.edf")}


def fetch_run_paths(subject: int, runs: list[int]) -> list[Path]:
    """Paths to a subject's EDF files: local folder first, MNE download otherwise.

    The fallback keeps `python main.py` working on a fresh machine with no
    manual download, which the brief requires.
    """
    paths = []
    for run in runs:
        local = _local_index().get(f"S{subject:03d}R{run:02d}.EDF")
        if local is None:
            # Positional subject argument: the keyword was renamed across MNE versions.
            (local,) = eegbci.load_data(subject, [run], update_path=False, verbose="error")
        paths.append(Path(local))
    return paths


def inspect_run(subject: int, run: int, path: Path) -> dict:
    """Everything the audit needs to know about one EDF run."""
    raw = mne.io.read_raw_edf(path, preload=True, verbose="error")
    sfreq = float(raw.info["sfreq"])
    duration = raw.n_times / sfreq

    desc = [str(d) for d in raw.annotations.description]
    onsets = np.asarray(raw.annotations.onset)
    durs = np.asarray(raw.annotations.duration)
    is_cue = np.isin(desc, ["T1", "T2"])

    # Signal checks on a 1 Hz high-passed copy: raw EDF can carry DC offsets
    # and slow drift that would make every channel look "huge".
    data = raw.copy().filter(1.0, None, verbose="error").get_data()
    sd = data.std(axis=1)
    ptp = np.ptp(data, axis=1)

    return {
        "subject": subject,
        "run": run,
        # Path relative to the data folder, so the CSV is identical on any machine.
        "file": (path.relative_to(DATA_DIR).as_posix() if path.is_relative_to(DATA_DIR) else path.name),
        "sfreq_hz": sfreq,
        "duration_s": round(duration, 2),
        "n_channels": len(raw.ch_names),
        "ch_names": tuple(raw.ch_names),
        "n_left": desc.count("T1"),
        "n_right": desc.count("T2"),
        "n_rest": desc.count("T0"),
        "cue_dur_min_s": round(float(durs[is_cue].min()), 3) if is_cue.any() else np.nan,
        "cue_dur_max_s": round(float(durs[is_cue].max()), 3) if is_cue.any() else np.nan,
        "cue_dur_median_s": round(float(np.median(durs[is_cue])), 3) if is_cue.any() else np.nan,
        "unknown_codes": sorted(set(desc) - set(EVENT_MAP)),
        # Events whose window runs past the end of the recording cannot be epoched.
        "n_events_past_end": int(np.sum(onsets + durs > duration + 1.0 / sfreq)),
        "flat_channels": [c for c, s in zip(raw.ch_names, sd) if s < FLAT_SD_V],
        "huge_channels": [c for c, p in zip(raw.ch_names, ptp) if p > HUGE_PTP_V],
    }


def _run_level_anomalies(r: dict, ref: dict) -> list[str]:
    """Anomalies of one run relative to the protocol and the dataset median."""
    out = []
    if r["sfreq_hz"] != NOMINAL_SFREQ:
        out.append(("sfreq_mismatch", f"run {r['run']}: {r['sfreq_hz']:g} Hz"))
    if r["n_channels"] != NOMINAL_N_CH or r["ch_names"] != ref["ch_names"]:
        out.append(("channel_mismatch", f"run {r['run']}: {r['n_channels']} ch / names differ"))
    if abs(r["duration_s"] - ref["duration_s"]) > RUN_DURATION_TOL_S:
        out.append(("run_duration", f"run {r['run']}: {r['duration_s']:.1f} s vs median {ref['duration_s']:.1f} s"))
    n_cues = r["n_left"] + r["n_right"]
    if n_cues != ref["n_cues"]:
        out.append(("cue_count", f"run {r['run']}: {n_cues} cues vs median {ref['n_cues']}"))
    # A different median cue length means a different protocol; a normal median
    # with one short cue means the recording stopped mid-trial.
    if np.isfinite(r["cue_dur_median_s"]) and abs(r["cue_dur_median_s"] - ref["cue_dur_s"]) > CUE_DURATION_TOL_S:
        out.append(("protocol_timing", f"run {r['run']}: cues {r['cue_dur_median_s']} s vs dataset {ref['cue_dur_s']} s"))
    elif np.isfinite(r["cue_dur_min_s"]) and ref["cue_dur_s"] - r["cue_dur_min_s"] > CUE_DURATION_TOL_S:
        out.append(("truncated_trial", f"run {r['run']}: shortest cue {r['cue_dur_min_s']} s (recording ends mid-trial)"))
    if r["unknown_codes"]:
        out.append(("unknown_event_codes", f"run {r['run']}: {r['unknown_codes']}"))
    if r["n_events_past_end"]:
        out.append(("events_past_end", f"run {r['run']}: {r['n_events_past_end']} events"))
    bad = sorted(set(r["flat_channels"]) | set(r["huge_channels"]))
    if bad:
        out.append(("high_amplitude_channels", f"run {r['run']}: flat={[c.strip('.') for c in r['flat_channels']]} >1mV={[c.strip('.') for c in r['huge_channels']]}"))
    return out


def _reference(runs: list[dict]) -> dict:
    """Dataset-wide medians that individual runs are compared against."""
    # The most common channel list is the reference montage, so one odd
    # subject cannot redefine what "normal" is.
    names, counts = np.unique([";".join(r["ch_names"]) for r in runs], return_counts=True)
    cue_durs = [r["cue_dur_median_s"] for r in runs if np.isfinite(r["cue_dur_median_s"])]
    return {
        "ch_names": tuple(names[np.argmax(counts)].split(";")),
        "duration_s": float(np.median([r["duration_s"] for r in runs])),
        "n_cues": int(np.median([r["n_left"] + r["n_right"] for r in runs])),
        "cue_dur_s": round(float(np.median(cue_durs)), 3),
    }


def run_audit(out_dir: Path) -> list[int]:
    """Audit all subjects, write the CSVs, return the subjects kept."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"reading EDFs from {DATA_DIR} ({len(_local_index())} found; missing ones are downloaded)")
    runs: list[dict] = []
    load_errors: dict[int, list[str]] = {}

    for subj in SUBJECTS:
        for run in IMAGERY_RUNS:
            try:
                (path,) = fetch_run_paths(subj, [run])
                runs.append(inspect_run(subj, run, path))
            except Exception as exc:  # a missing/corrupt file is an audit finding, not a crash
                load_errors.setdefault(subj, []).append(f"run {run}: {type(exc).__name__}: {exc}")
        print(f"audited subject {subj}")

    ref = _reference(runs)
    print(f"reference: {ref['duration_s']:.1f} s runs, {ref['n_cues']} cues/run, "
          f"{ref['cue_dur_s']} s cues, {len(ref['ch_names'])} channels")

    _write_run_table(runs, out_dir / "audit_runs.csv")

    subject_rows, kept = [], []
    for subj in SUBJECTS:
        sruns = [r for r in runs if r["subject"] == subj]
        anomalies = [a for r in sruns for a in _run_level_anomalies(r, ref)]
        if subj in load_errors or len(sruns) < len(IMAGERY_RUNS):
            anomalies.append(("missing_run", "; ".join(load_errors.get(subj, ["run(s) missing"]))))

        n_left = sum(r["n_left"] for r in sruns)
        n_right = sum(r["n_right"] for r in sruns)
        if min(n_left, n_right) < MIN_TRIALS_PER_CLASS:
            anomalies.append(("too_few_trials", f"L={n_left} R={n_right} < {MIN_TRIALS_PER_CLASS}"))
        if n_left + n_right and abs(n_left - n_right) / (n_left + n_right) > IMBALANCE_FRAC:
            anomalies.append(("class_imbalance", f"L={n_left} R={n_right}"))

        kinds = sorted({k for k, _ in anomalies})
        action = max((ACTION_POLICY[k] for k in kinds), key=_SEVERITY.get, default="keep")
        reason = "; ".join(f"{k}->{ACTION_POLICY[k]}" for k in kinds) or "no anomalies"
        if action != "drop":
            kept.append(subj)

        subject_rows.append({
            "subject": subj,
            "runs_loaded": " ".join(str(r["run"]) for r in sruns),
            "sfreq_hz": " ".join(sorted({f"{r['sfreq_hz']:g}" for r in sruns})),
            "run_durations_s": " ".join(f"{r['duration_s']:.1f}" for r in sruns),
            "n_channels": " ".join(sorted({str(r["n_channels"]) for r in sruns})),
            "n_left": n_left,
            "n_right": n_right,
            "n_rest": sum(r["n_rest"] for r in sruns),
            "left_right_per_run": " ".join(f"{r['run']}:{r['n_left']}/{r['n_right']}" for r in sruns),
            "cue_dur_range_s": (f"{min(r['cue_dur_min_s'] for r in sruns)}-{max(r['cue_dur_max_s'] for r in sruns)}"
                                if sruns else ""),
            "event_mapping": "runs 4/8/12: T0=rest, T1=imagined left fist, T2=imagined right fist",
            "anomalies": " | ".join(msg for _, msg in anomalies) or "none",
            "action": action,
            "action_reason": reason,
        })

    _write_csv(subject_rows, out_dir / "audit.csv")
    dropped = [r["subject"] for r in subject_rows if r["action"] == "drop"]
    resampled = [r["subject"] for r in subject_rows if r["action"] == "resample"]
    print(f"\naudit.csv written: {len(kept)} kept, {len(dropped)} dropped {dropped}, "
          f"{len(resampled)} to resample {resampled}")
    return kept


def _write_run_table(runs: list[dict], path: Path) -> None:
    rows = []
    for r in runs:
        row = {k: v for k, v in r.items() if k != "ch_names"}
        for k in ("unknown_codes", "flat_channels", "huge_channels"):
            row[k] = " ".join(c.strip(".") for c in row[k]) or "none"
        rows.append(row)
    _write_csv(rows, path)


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)