"""Shared trial loader for Task 1 (A2, A3 and Part B all use these epochs).

Pipeline per run: read EDF -> standard channel names -> montage -> interpolate the
channels the A1 audit flagged in *that run* -> band-pass -> epoch around T1/T2 cues.
Epochs are cached in cache/epochs/ so later stages and re-runs do not reload EDFs.
"""
from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd

from task1.data_audit import IMAGERY_RUNS, TRIAL_WINDOW, load_run

ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "outputs" / "task1"
CACHE = ROOT / "cache" / "epochs"

# 1-40 Hz: keeps theta through beta (all bands any later analysis uses), removes slow
# drift that would dominate low-frequency power, and removes 60 Hz mains (the data were
# recorded in the US, BCI2000/Wadsworth). The A3 ocular decoder needs slow activity, so
# it loads with its own low cut via `l_freq`.
L_FREQ, H_FREQ = 1.0, 40.0
TMIN, TMAX = TRIAL_WINDOW
EVENT_ID = {"left": 1, "right": 2}  # T1 = left fist, T2 = right fist (see data_audit)
LABEL = {1: 0, 2: 1}                # class index used by decoders and networks


def kept_subjects() -> list[int]:
    df = pd.read_csv(AUDIT_DIR / "audit.csv")
    return df.loc[df.action != "exclude", "subject"].astype(int).tolist()


def _run_bad_channels() -> dict[tuple[int, int], list[str]]:
    """Bad channels per (subject, run), from the per-run audit rather than the subject
    union, so a channel that was fine in one run is not needlessly interpolated there."""
    runs = pd.read_csv(AUDIT_DIR / "audit_runs.csv").fillna("")
    out = {}
    for r in runs.itertuples():
        chans = [c for f in r.anomalies.split(";") if f.startswith(("flat_ch_", "noisy_ch_"))
                 for c in f.split("_", 2)[2].split("/")]
        out[(int(r.subject), int(r.run))] = chans
    return out


def load_epochs(subject: int, l_freq: float = L_FREQ, h_freq: float = H_FREQ,
                use_cache: bool = True) -> mne.Epochs:
    tag = f"S{subject:03d}_{l_freq:g}-{h_freq:g}Hz-epo.fif"
    path = CACHE / tag
    if use_cache and path.exists():
        return mne.read_epochs(path, preload=True, verbose="error")

    bads = _run_bad_channels()
    montage = mne.channels.make_standard_montage("standard_1005")
    per_run = []
    for run in IMAGERY_RUNS:
        raw = load_run(subject, run)
        raw.set_montage(montage, verbose="error")
        raw.info["bads"] = [c for c in bads.get((subject, run), []) if c in raw.ch_names]
        if raw.info["bads"]:
            # Spherical-spline interpolation: keeps all 64 channels, which the CSD in A2
            # and the fixed-size network input in Part B both need.
            raw.interpolate_bads(reset_bads=True, verbose="error")
        # Filter each run separately, before epoching: filtering across run boundaries
        # would smear the discontinuity, and filtering short epochs creates edge artefacts.
        raw.filter(l_freq, h_freq, fir_design="firwin", verbose="error")
        events, _ = mne.events_from_annotations(raw, event_id={"T1": 1, "T2": 2}, verbose="error")
        ep = mne.Epochs(raw, events, EVENT_ID, tmin=TMIN, tmax=TMAX, baseline=None,
                        preload=True, verbose="error")
        ep.metadata = pd.DataFrame({"subject": subject, "run": run,
                                    "onset_s": ep.events[:, 0] / raw.info["sfreq"]})
        per_run.append(ep)
    epochs = mne.concatenate_epochs(per_run, verbose="error")

    CACHE.mkdir(parents=True, exist_ok=True)
    epochs.save(path, overwrite=True, fmt="single", verbose="error")
    return epochs
