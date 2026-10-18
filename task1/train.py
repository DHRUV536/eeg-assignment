"""Part B harness: fixed subject-wise split, train-only normalisation, training with early
stopping on validation subjects, resumable runs, and test evaluation per subject.

Cross-subject rule: test subjects never touch training, normalisation statistics, channel
selection or early stopping. Validation subjects are used only for early stopping.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from task1.data import CACHE, kept_subjects, load_epochs
from task1.models import EEGNet, N_T, build, n_params

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "task1"
MODEL_DIR = CACHE.parent / "models"
SEED = 42
N_TEST, N_VAL = 10, 4          # -> 23 training subjects (see DECISIONS)
BATCH, LR, WD = 64, 1e-3, 1e-3
MAX_EPOCHS, PATIENCE = 150, 25
TMIN = 0.0                     # network input: 0-4 s after the cue (640 samples at 160 Hz)


# ---------------------------------------------------------------- data and split
def network_data() -> dict:
    """All kept trials, 1-40 Hz, cropped to 0-4 s, float32 (cached as one .npz)."""
    path = CACHE.parent / "b_data.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    X, y, subj, ch = [], [], [], None
    for s in kept_subjects():
        ep = load_epochs(s)
        i0 = int(np.searchsorted(ep.times, TMIN - 1e-9))
        X.append(ep.get_data()[:, :, i0:i0 + N_T].astype(np.float32))
        y.append((ep.events[:, 2] == 2).astype(np.int64))
        subj.append(np.full(len(ep), s))
        ch = ep.ch_names
    d = dict(X=np.concatenate(X), y=np.concatenate(y), subject=np.concatenate(subj),
             ch_names=np.array(ch))
    tmp = CACHE.parent / f"tmp_{os.getpid()}_b_data.npz"
    np.savez(tmp, **d)
    os.replace(tmp, path)
    return d


def subject_split() -> dict[str, list[int]]:
    """Fixed, seeded random split of the 37 kept subjects. Random rather than chosen on any
    Part A result, so the split cannot favour (or hide) the confound or the ERD."""
    subs = np.array(kept_subjects())
    perm = np.random.default_rng(SEED).permutation(subs)
    split = {"test": sorted(perm[:N_TEST].tolist()),
             "val": sorted(perm[N_TEST:N_TEST + N_VAL].tolist()),
             # Training order is the permutation order, so "the first n" nested subsets
             # (5 subset of 20 subset of all) are random and fixed.
             "train": perm[N_TEST + N_VAL:].tolist()}
    rows = [dict(subject=s, set=k, train_order=(split["train"].index(s) if k == "train" else ""))
            for k, v in split.items() for s in v]
    pd.DataFrame(rows).sort_values("subject").to_csv(OUT / "split.csv", index=False)
    return split


class Normaliser:
    """Per-channel z-score with statistics from the training subjects only."""

    def __init__(self, X_train: np.ndarray):
        self.mu = X_train.mean(axis=(0, 2), keepdims=True)
        self.sd = X_train.std(axis=(0, 2), keepdims=True) + 1e-12

    def __call__(self, X):
        return ((X - self.mu) / self.sd).astype(np.float32)


def prepare(train_subjects, split=None, data=None):
    data = data if data is not None else network_data()
    split = split or subject_split()
    pick = lambda subs: np.isin(data["subject"], subs)
    tr, va, te = pick(train_subjects), pick(split["val"]), pick(split["test"])
    norm = Normaliser(data["X"][tr])
    return dict(Xtr=norm(data["X"][tr]), ytr=data["y"][tr],
                Xva=norm(data["X"][va]), yva=data["y"][va],
                Xte=norm(data["X"][te]), yte=data["y"][te], ste=data["subject"][te],
                ch_names=data["ch_names"])


# ---------------------------------------------------------------- training
@torch.no_grad()
def predict_proba(model, X, batch=256):
    model.eval()
    out = [torch.softmax(model(torch.from_numpy(X[i:i + batch])), 1).numpy()
           for i in range(0, len(X), batch)]
    return np.concatenate(out)


def _atomic_save(obj, path: Path):
    tmp = path.with_name(f"tmp_{os.getpid()}_{path.name}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train_run(run_id: str, model_name: str, tokeniser: str | None, train_subjects,
              seed: int = SEED, deadline: float | None = None) -> dict | None:
    """Train one model; resumes from its per-epoch checkpoint if one exists. Returns the
    result dict when finished, or None if `deadline` (time.time()) was reached first."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    final, ckpt = MODEL_DIR / f"{run_id}.pt", MODEL_DIR / f"{run_id}.ckpt"
    D = prepare(train_subjects)

    torch.manual_seed(seed)
    model = build(model_name, tokeniser)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    gen = torch.Generator().manual_seed(seed)          # batch order
    state = dict(epoch=0, best_val=np.inf, best_epoch=-1, bad=0, best_state=None, history=[])
    if ckpt.exists():
        c = torch.load(ckpt, weights_only=False)
        model.load_state_dict(c["model"])
        opt.load_state_dict(c["opt"])
        gen.set_state(c["gen"])
        torch.set_rng_state(c["torch_rng"])              # dropout stream
        state = c["state"]

    Xtr, ytr = torch.from_numpy(D["Xtr"]), torch.from_numpy(D["ytr"])
    loss_fn = nn.CrossEntropyLoss()
    while state["epoch"] < MAX_EPOCHS and state["bad"] < PATIENCE:
        model.train()
        perm = torch.randperm(len(Xtr), generator=gen)
        tot = 0.0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            if isinstance(model, EEGNet):
                model.apply_max_norm()
            tot += loss.item() * len(idx)
        p = predict_proba(model, D["Xva"])
        vloss = float(-np.log(p[np.arange(len(p)), D["yva"]] + 1e-12).mean())
        vacc = float((p.argmax(1) == D["yva"]).mean())
        state["history"].append(dict(epoch=state["epoch"], train_loss=tot / len(perm),
                                     val_loss=vloss, val_acc=vacc))
        # Early stopping on validation loss (smoother than accuracy on 4 subjects).
        if vloss < state["best_val"] - 1e-4:
            state.update(best_val=vloss, best_epoch=state["epoch"], bad=0,
                         best_state={k: v.clone() for k, v in model.state_dict().items()})
        else:
            state["bad"] += 1
        state["epoch"] += 1
        _atomic_save(dict(model=model.state_dict(), opt=opt.state_dict(), gen=gen.get_state(),
                          torch_rng=torch.get_rng_state(), state=state), ckpt)
        if deadline is not None and time.time() > deadline:
            return None

    model.load_state_dict(state["best_state"])
    _atomic_save(model.state_dict(), final)
    pte = predict_proba(model, D["Xte"])
    pva = predict_proba(model, D["Xva"])
    per_subj = {int(s): float((pte[D["ste"] == s].argmax(1) == D["yte"][D["ste"] == s]).mean())
                for s in np.unique(D["ste"])}
    hist = pd.DataFrame(state["history"])
    hist.to_csv(MODEL_DIR / f"{run_id}_history.csv", index=False)
    return dict(run_id=run_id, model=model_name, tokeniser=tokeniser or "",
                n_train_subjects=len(train_subjects), n_train_trials=len(ytr), seed=seed,
                n_params=n_params(model), epochs_run=state["epoch"],
                best_epoch=state["best_epoch"], val_acc=float((pva.argmax(1) == D["yva"]).mean()),
                test_acc=float((pte.argmax(1) == D["yte"]).mean()),
                test_acc_mean_over_subjects=float(np.mean(list(per_subj.values()))),
                per_subject=json.dumps(per_subj))


def load_trained(run_id: str, model_name: str, tokeniser: str | None) -> nn.Module:
    m = build(model_name, tokeniser)
    m.load_state_dict(torch.load(MODEL_DIR / f"{run_id}.pt", weights_only=True))
    return m.eval()


def run_all(runs: list[dict], csv_path: Path, budget_s: float | None = None) -> pd.DataFrame:
    """Train every run not yet in csv_path. With a time budget (EEG_TIME_BUDGET seconds) the
    loop stops cleanly and the next invocation resumes; without one it runs to the end."""
    deadline = time.time() + budget_s if budget_s else None
    done = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
    for r in runs:
        if len(done) and r["run_id"] in set(done.run_id):
            continue
        res = train_run(deadline=deadline, **r)
        if res is None:
            print(f"[B] {r['run_id']}: time budget reached, will resume")
            return done
        done = pd.concat([done, pd.DataFrame([res])], ignore_index=True)
        done.to_csv(csv_path, index=False)
        print(f"[B] {res['run_id']}: test acc {res['test_acc']:.3f} "
              f"(val {res['val_acc']:.3f}, best epoch {res['best_epoch']})", flush=True)
    return done
