"""
Shared trial loader for A2, A3 and Part B.

One function, `load_subject`, turns a subject's three imagery runs into
cue-locked trials. Everything downstream (ERD, confound decoders, networks)
starts from the same trials, so a trial rejected here is rejected everywhere
and the per-subject counts in the results table are the real ones.
"""
from __future__ import annotations

import csv
from pathlib import Path

import mne
import numpy as np
from mne.datasets import eegbci

from task1.audit import IMAGERY_RUNS, fetch_run_paths

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "cache" / "epochs"

# Cue-locked window. 2 s before the cue covers the rest (T0) period that
# precedes every cue (4.1 s long, verified in audit_runs.csv), so the A2
# baseline can sit inside it. The end is 3.975 s, not 4.0 s: in many runs the
# last cue starts at 119.0 s and the file ends at 122.994 s, so a 4.0 s window
# would silently drop that trial in 6 subjects (found by counting events).
TMIN, TMAX = -2.0, 3.975
SFREQ = 160.0

# Line noise: the recordings were made in the USA (60 Hz mains). Checked on
# the data itself in DECISIONS.md D6 (PSD peak at 60 Hz).
LINE_FREQ = 60.0
# 0.5 Hz high-pass removes electrode drift but keeps the slow (<4 Hz) ocular
# signal that A3 must be able to see; nothing above 0.5 Hz is removed so the
# EMG route (>30 Hz) also stays testable. Analyses band-limit further themselves.
HP_FREQ = 0.5

# Gross-artifact rejection, relative to each subject's own trials.
# A fixed microvolt threshold does not work here: amplitude differs ~5-8x
# between subjects on every channel (S072 C3 ~60 uV p-p vs S075 ~530 uV),
# largely a common-mode signal from the undocumented reference (DECISIONS D8, D9).
# A trial is rejected if its largest non-ocular peak-to-peak exceeds
# REJECT_FACTOR x that subject's median, i.e. an outlier for that recording.
# Ocular channels (Fp/AF) are excluded from the test on purpose: rejecting
# trials on eye activity would remove exactly the confound A3 has to measure.
REJECT_FACTOR = 3.0
OCULAR_PREFIXES = ("Fp", "AF")

LABELS = {"T1": 0, "T2": 1}          # 0 = imagined left fist, 1 = imagined right fist
CLASS_NAMES = ["left", "right"]


def kept_subjects(audit_csv: Path) -> list[int]:
    with open(audit_csv) as f:
        return [int(r["subject"]) for r in csv.DictReader(f) if r["action"] != "drop"]


def _load_run(subject: int, run: int) -> mne.io.BaseRaw:
    (path,) = fetch_run_paths(subject, [run])
    raw = mne.io.read_raw_edf(path, preload=True, verbose="error")
    eegbci.standardize(raw)                       # "C3.." -> "C3"
    raw.set_montage("standard_1005", verbose="error")
    # Filter each run separately so filter edges never straddle the gap
    # between two recordings.
    raw.notch_filter(LINE_FREQ, verbose="error")
    raw.filter(HP_FREQ, None, verbose="error")
    return raw


def load_subject(subject: int, use_cache: bool = True) -> dict:
    """Cue-locked trials for one subject.

    Returns dict with X (n_trials, 64, n_times) float32 in microvolts,
    y (0 left / 1 right), run, trial_in_run, onset_s, ch_names, times,
    n_rejected and n_events.
    """
    cache_file = CACHE / f"S{subject:03d}.npz"
    if use_cache and cache_file.exists():
        d = np.load(cache_file, allow_pickle=False)
        return {k: d[k] for k in d.files}

    X, y, run_ids, idx_in_run, onsets = [], [], [], [], []
    n_events = 0
    ch_names = times = None
    for run in IMAGERY_RUNS:
        raw = _load_run(subject, run)
        events, _ = mne.events_from_annotations(
            raw, event_id={"T1": 1, "T2": 2}, verbose="error")
        # MNE silently drops trials whose window runs past the end of the
        # recording (S104 run 8); counting events first makes that visible.
        n_events += len(events)
        ep = mne.Epochs(raw, events, {"T1": 1, "T2": 2}, tmin=TMIN, tmax=TMAX,
                        baseline=None, preload=True, reject=None, verbose="error")
        X.append(ep.get_data(units="uV"))
        y.append((ep.events[:, 2] - 1).astype(np.int64))       # T1 -> 0, T2 -> 1
        run_ids.append(np.full(len(ep), run))
        # position of each trial in its run: lets A3 test order/time effects
        idx_in_run.append(np.arange(len(ep)))
        onsets.append(ep.events[:, 0] / raw.info["sfreq"])
        ch_names, times = ep.ch_names, ep.times

    X = np.concatenate(X)
    non_ocular = [i for i, c in enumerate(ch_names) if not c.startswith(OCULAR_PREFIXES)]
    ptp = np.ptp(X[:, non_ocular], axis=-1).max(axis=1)
    good = ptp <= REJECT_FACTOR * np.median(ptp)

    out = {
        "X": X[good].astype(np.float32),
        "y": np.concatenate(y)[good],
        "run": np.concatenate(run_ids)[good],
        "trial_in_run": np.concatenate(idx_in_run)[good],
        "onset_s": np.concatenate(onsets)[good],
        "ch_names": np.array(ch_names),
        "times": times,
        "n_events": np.array(n_events),
        "n_epoched": np.array(len(X)),
        "n_rejected": np.array(int((~good).sum())),
        "subject": np.array(subject),
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(cache_file, **out)
    return out


def channel_positions(ch_names) -> np.ndarray:
    """3-D positions (m) of the standard_1005 electrodes, in ch_names order."""
    pos = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    return np.array([pos[c] for c in ch_names])
