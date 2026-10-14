"""
Part B training and evaluation harness (cross-subject only).

Leakage rules (brief, Part B), and how they are met here:
  * Subject-wise folds: 5 fixed folds of the 37 kept subjects (folds.csv).
    A test subject's trials are never in the training set.
  * Normalisation: every step of preprocessing is either a fixed linear map
    (CAR, resampling) or uses statistics of the single trial being processed
    (per-trial scale). No statistic is ever pooled over trials, so nothing
    can flow from test trials into training or from one test trial to another.
  * Channel selection: none; all 64 channels are always used.
  * Early stopping: none. Every run trains for a fixed number of steps set
    in advance from the training-set size (steps_for).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from task1.data import kept_subjects, load_subject
from task1.models import N_TIMES, SFREQ as NET_SFREQ, build_model, n_params

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
N_FOLDS = 5
SEEDS = (0, 1, 2)                 # offsets added to the project seed
BATCH = 64
LR, WD = 1e-3, 1e-2
WARMUP_FRAC = 0.05
torch.set_num_threads(2)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def preprocess(X160: np.ndarray, times160: np.ndarray) -> np.ndarray:
    """Raw-reference trials at 160 Hz (n, 64, t, microvolts) -> network input.

    1. crop 0 to 3.975 s after the cue: the network sees the imagery period
       only; the pre-cue window carries eye position from the previous trial
       (A3, R5_precue_ocular), which is not motor imagery.
    2. common average reference: removes the large signal shared by all
       channels (25-60% of variance, DECISIONS D9), a fixed linear map.
    3. resample 160 -> 80 Hz (polyphase with anti-alias filter).
    4. divide each trial by its own SD over channels and time: subjects differ
       5-8x in amplitude (D8); a per-trial scalar keeps every within-trial
       relation (C3 vs C4, frontal vs central) and uses no other trial.
    """
    X = X160[..., times160 >= 0.0]
    X = X - X.mean(axis=1, keepdims=True)
    X = mne.filter.resample(X.astype(np.float64), down=160 / NET_SFREQ, axis=-1, verbose="error")
    X = X[..., :N_TIMES]
    X = X / X.std(axis=(1, 2), keepdims=True)
    return X.astype(np.float32)


# B2.5: the ocular route removed. Channels on the forehead and lateral-frontal
# sites where A3 found the eye signal (a3_channel_topography.csv), plus T9/T10
# below them. Everything from the FC row backwards is kept.
OCULAR_SITES = ["Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8", "F7", "F5", "F3", "F1",
                "Fz", "F2", "F4", "F6", "F8", "FT7", "FT8", "T9", "T10"]
MOTOR_BAND = (8.0, 30.0)
ERD_BASELINE = (-1.5, -0.5)


def preprocess_variant(X160: np.ndarray, times160: np.ndarray, ch_names, variant: str) -> np.ndarray:
    """Network input with the ocular route removed (B2.5) and, optionally,
    each channel expressed relative to its own pre-cue power (B2.6).

    no_ocular:   drop OCULAR_SITES, CAR over the remaining 43 channels (so the
                 average no longer carries frontal EOG into every channel),
                 band-pass 8-30 Hz (removes the < 4 Hz saccade/eye-position
                 signal and most of the visual evoked response's power), then
                 crop / resample / per-trial scale as in `preprocess`. Dropped
                 channels are fed as zeros so the architectures are unchanged.
    no_ocular_erdnorm: as no_ocular, but each channel is divided by its own
                 8-30 Hz RMS in the pre-cue baseline (-1.5 to -0.5 s) of the
                 SAME trial instead of the trial-wide scalar (see D29).
    """
    from scipy.signal import butter, sosfiltfilt
    keep = np.array([c not in OCULAR_SITES for c in ch_names])
    X = X160.astype(np.float64).copy()
    X[:, keep] -= X[:, keep].mean(axis=1, keepdims=True)
    X[:, ~keep] = 0.0
    sos = butter(4, MOTOR_BAND, btype="bandpass", fs=160.0, output="sos")
    # filter the whole -2..4 s trial, then crop: keeps filter edges out of the window
    X[:, keep] = sosfiltfilt(sos, X[:, keep], axis=-1)
    if variant == "no_ocular_erdnorm":
        b = (times160 >= ERD_BASELINE[0]) & (times160 <= ERD_BASELINE[1])
        rms = np.sqrt((X[:, keep][..., b] ** 2).mean(-1, keepdims=True))
        X[:, keep] = X[:, keep] / rms
    X = X[..., times160 >= 0.0]
    X = mne.filter.resample(X, down=160 / NET_SFREQ, axis=-1, verbose="error")[..., :N_TIMES]
    if variant == "no_ocular":
        X = X / X[:, keep].std(axis=(1, 2), keepdims=True)
    return X.astype(np.float32)


def load_net_data(audit_csv: Path, variant: str = "default") -> dict:
    cache = CACHE / ("net_inputs.npz" if variant == "default" else f"net_inputs_{variant}.npz")
    if cache.exists():
        d = np.load(cache)
        return {k: d[k] for k in d.files}
    Xs, ys, ss = [], [], []
    for s in kept_subjects(audit_csv):
        d = load_subject(s)
        X = (preprocess(d["X"], d["times"]) if variant == "default"
             else preprocess_variant(d["X"], d["times"], list(d["ch_names"]), variant))
        Xs.append(X); ys.append(d["y"]); ss.append(np.full(len(d["y"]), s))
    out = {"X": np.concatenate(Xs), "y": np.concatenate(ys), "subject": np.concatenate(ss),
           "ch_names": d["ch_names"]}
    CACHE.mkdir(exist_ok=True)
    np.savez(cache, **out)
    return out


def make_folds(subjects, seed: int) -> dict[int, int]:
    """Fixed subject -> fold map: one seeded permutation split into 5 folds."""
    perm = np.random.default_rng(seed).permutation(sorted(subjects))
    return {int(s): f for f, part in enumerate(np.array_split(perm, N_FOLDS)) for s in part}


def training_subjects(pool, n, fold, seed_offset, project_seed):
    """Nested subsets: one permutation per (fold, seed), first n subjects.
    5 ⊂ 10 ⊂ 20 ⊂ all, and both models get the same subjects."""
    if n is None or n >= len(pool):
        return sorted(pool)
    rng = np.random.default_rng(project_seed + 1000 * (fold + 1) + seed_offset)
    return sorted(rng.permutation(sorted(pool))[:n].tolist())


def steps_for(n_trials: int) -> int:
    """Fixed training length from training-set size only: 25 passes over the
    data, at least 200 and at most 500 steps of 64 trials."""
    return int(np.clip(round(25 * n_trials / BATCH), 200, 500))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(model_name: str, X: np.ndarray, y: np.ndarray, seed: int, steps: int | None = None,
          augment=None, log_every: int = 0):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = build_model(model_name)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps = steps or steps_for(len(y))
    warm = max(1, int(WARMUP_FRAC * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: min(1.0, (k + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1.0, k / steps))))
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    model.train()
    losses = []
    for k in range(steps):
        idx = torch.randint(0, len(yt), (BATCH,), generator=gen)
        xb, yb = Xt[idx], yt[idx]
        if augment is not None:
            xb, yb = augment(xb, yb, gen)
        loss = F.cross_entropy(model(xb), yb)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        model.apply_max_norm()
        losses.append(loss.item())
        if log_every and (k + 1) % log_every == 0:
            print(f"    step {k + 1}/{steps} loss {np.mean(losses[-log_every:]):.3f}", flush=True)
    model.eval()
    return model, float(np.mean(losses[-50:]))


@torch.no_grad()
def predict_proba(model, X: np.ndarray, batch: int = 256) -> np.ndarray:
    model.eval()
    out = [F.softmax(model(torch.from_numpy(X[i:i + batch])), dim=1)[:, 1].numpy()
           for i in range(0, len(X), batch)]
    return np.concatenate(out)


def per_subject_accuracy(p: np.ndarray, y: np.ndarray, subj: np.ndarray) -> dict:
    pred = (p >= 0.5).astype(int)
    return {int(s): float((pred[subj == s] == y[subj == s]).mean()) for s in np.unique(subj)}


# ---------------------------------------------------------------------------
# Experiment runner with resumable results
# ---------------------------------------------------------------------------
class Runner:
    """Runs (experiment, model, fold, n_train, seed) jobs once, appends one
    row per test subject to a CSV, and saves weights for later analysis.
    Re-running skips finished jobs, so an interrupted main.py resumes."""

    def __init__(self, out_csv: Path, data: dict, folds: dict, project_seed: int):
        self.out_csv, self.d, self.folds, self.seed = out_csv, data, folds, project_seed
        self.done = set()
        if out_csv.exists():
            prev = pd.read_csv(out_csv)
            self.done = set(prev["job"].unique())
        (CACHE / "models").mkdir(parents=True, exist_ok=True)

    def run(self, experiment, model_name, fold, n_train, seed_offset, augment=None,
            transform=None, tag="", save_weights=False):
        job = f"{experiment}|{model_name}{tag}|f{fold}|n{n_train or 'all'}|s{seed_offset}"
        if job in self.done:
            return
        d = self.d
        subj = d["subject"]
        test_mask = np.array([self.folds[int(s)] == fold for s in subj])
        pool = sorted({int(s) for s in subj[~test_mask]})
        train_subj = training_subjects(pool, n_train, fold, seed_offset, self.seed)
        tr = np.isin(subj, train_subj)
        Xtr, Xte = d["X"][tr], d["X"][test_mask]
        if transform is not None:            # e.g. B2.5 confound removal; same map on train and test
            Xtr, Xte = transform(Xtr), transform(Xte)
        run_seed = self.seed + 100 * seed_offset + fold
        t0 = time.time()
        model, final_loss = train(model_name, Xtr, d["y"][tr], run_seed, augment=augment)
        p = predict_proba(model, Xte)
        accs = per_subject_accuracy(p, d["y"][test_mask], subj[test_mask])
        rows = [{"job": job, "experiment": experiment, "model": model_name + tag, "fold": fold,
                 "n_train_subjects": len(train_subj), "seed": seed_offset, "run_seed": run_seed,
                 "subject": s, "acc": a, "n_test_trials": int((subj == s).sum()),
                 "final_train_loss": final_loss, "n_params": n_params(model),
                 "train_subjects": " ".join(map(str, train_subj))} for s, a in accs.items()]
        df = pd.DataFrame(rows)
        df.to_csv(self.out_csv, mode="a", header=not self.out_csv.exists(), index=False, float_format="%.5f")
        self.done.add(job)
        if save_weights:
            torch.save(model.state_dict(), CACHE / "models" / (job.replace("|", "__") + ".pt"))
        mean_acc = np.mean([(p[subj[test_mask] == s] >= 0.5).astype(int).__eq__(d["y"][test_mask][subj[test_mask] == s]).mean()
                            for s in accs])
        print(f"  {job}: mean subject acc {mean_acc:.3f}, train loss {final_loss:.3f}, "
              f"{time.time() - t0:.0f}s", flush=True)


def load_trained(job: str, model_name: str):
    m = build_model(model_name)
    m.load_state_dict(torch.load(CACHE / "models" / (job.replace("|", "__") + ".pt")))
    m.eval()
    return m
