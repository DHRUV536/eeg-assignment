"""
B2.3 - Electrode displacement at test time only.

A whole-cap misplacement is simulated as a rigid rotation of all 64
electrode positions on the head sphere:
  lateral   rotation about the anterior-posterior (nasion-inion) axis:
            the cap slides towards the left or right ear
  AP        rotation about the left-right (ear-ear) axis:
            the cap slides towards the nose or the back of the head
The signal each electrode would record at its displaced position is
interpolated from the recorded signals at the TRUE positions with spherical
splines (Perrin et al. 1989), i.e. a 64 x 64 linear map applied to the raw
trials before the usual network preprocessing. Models are never retrained.

Size: 2.5, 5, 10, 15 degrees = 3.9, 7.9, 15.7, 23.6 mm of arc at the vertex
for a 9 cm head radius. Normal placement varies with SD of 3.9 mm (left-right),
5.6 mm (front-back) and 7.2 mm (vertical) across 20 participants
(Brain and Behavior 2022, "Variability of EEG electrode positions and their
underlying brain regions"), so the range spans 0 to ~4 SD, from routine to
clearly careless placement.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.polynomial.legendre import legval

from task1.b2_experiments import ALL_MODELS
from task1.data import channel_positions, kept_subjects, load_subject
from task1.plotting import C, MODEL_COLORS, plt, style
from task1.train import (N_FOLDS, SEEDS, load_net_data, load_trained, make_folds,
                         predict_proba, preprocess)

ANGLES = [2.5, 5.0, 10.0, 15.0]
DIRECTIONS = {"right": ("y", +1), "left": ("y", -1), "anterior": ("x", +1), "posterior": ("x", -1)}
HEAD_RADIUS_MM = 90.0


# ---------------------------------------------------------------------------
# Geometry and spherical-spline interpolation
# ---------------------------------------------------------------------------
def unit_positions(ch_names):
    """Electrode directions from the best-fitting sphere's centre."""
    P = channel_positions(ch_names)
    # Least-squares sphere fit: |p - c|^2 = r^2  ->  linear in (c, r^2 - |c|^2)
    A = np.c_[2 * P, np.ones(len(P))]
    sol, *_ = np.linalg.lstsq(A, (P ** 2).sum(1), rcond=None)
    centre = sol[:3]
    U = P - centre
    return U / np.linalg.norm(U, axis=1, keepdims=True), centre


def rotation(axis: str, deg: float) -> np.ndarray:
    a = np.deg2rad(deg); c, s = np.cos(a), np.sin(a)
    # MNE head coordinates: x -> right ear, y -> nasion, z -> vertex.
    # Rotation about y moves the vertex along x (towards an ear): "lateral".
    # Rotation about x moves the vertex along y (towards nose/back): "AP".
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    raise ValueError(axis)


def _g(cos_angle, m=4, n_terms=50):
    n = np.arange(1, n_terms + 1)
    coef = np.r_[0.0, (2 * n + 1) / (n * (n + 1)) ** m] / (4 * np.pi)
    return legval(np.clip(cos_angle, -1, 1), coef)


# Regularisation: MNE's default (1e-5) is meant for repairing bad channels.
# On a full 64-channel grid it smooths so much that a ZERO-degree
# "displacement" already changes the data (each channel keeps 39% of its own
# signal; max |M - I| = 0.67). 1e-9 makes the 0-degree map the identity
# (max |M - I| = 0.004) and is still stable: worst-case channel gain 1.86 at
# 10 degrees (edge electrodes that rotate beyond the cap), signal power
# preserved within 3%. Checked in DECISIONS D27.
SPLINE_LAMBDA = 1e-9


def spline_matrix(U_from: np.ndarray, U_to: np.ndarray, lam: float = SPLINE_LAMBDA) -> np.ndarray:
    """Linear map from potentials at U_from to potentials at U_to (Perrin 1989).
    Constants are reproduced exactly (the c0 term), so the unknown reference,
    which adds the same value to every channel, passes through unchanged."""
    n = len(U_from)
    G = _g(U_from @ U_from.T) + lam * np.eye(n)
    A = np.block([[G, np.ones((n, 1))], [np.ones((1, n)), np.zeros((1, 1))]])
    Gt = np.c_[_g(U_to @ U_from.T), np.ones(len(U_to))]
    return Gt @ np.linalg.inv(A)[:, :n]


def displacement_matrix(ch_names, direction: str, deg: float) -> np.ndarray:
    U, _ = unit_positions(ch_names)
    axis, sign = DIRECTIONS[direction]
    U_new = U @ rotation(axis, sign * deg).T
    return spline_matrix(U, U_new)


def displacement_mm(ch_names, direction, deg) -> dict:
    U, _ = unit_positions(ch_names)
    axis, sign = DIRECTIONS[direction]
    U_new = U @ rotation(axis, sign * deg).T
    arc = np.arccos(np.clip((U * U_new).sum(1), -1, 1)) * HEAD_RADIUS_MM
    names = list(ch_names)
    return {"Cz": arc[names.index("Cz")], "C3": arc[names.index("C3")], "mean": arc.mean()}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def displaced_inputs(M: np.ndarray, subjects) -> np.ndarray:
    Xs = []
    for s in subjects:
        d = load_subject(s)
        Xd = np.einsum("ij,njt->nit", M, d["X"]).astype(np.float32)
        Xs.append(preprocess(Xd, d["times"]))
    return np.concatenate(Xs)


def run_displacement(out_dir: Path, seed: int) -> pd.DataFrame:
    data = load_net_data(out_dir / "audit.csv")
    subjects = kept_subjects(out_dir / "audit.csv")
    folds = make_folds(np.unique(data["subject"]), seed)
    ch = data["ch_names"]
    y, subj = data["y"], data["subject"]
    fold_of = np.array([folds[int(s)] for s in subj])

    models = {(m, f, s): load_trained(f"main|{m}|f{f}|nall|s{s}", m)
              for m in ALL_MODELS for f in range(N_FOLDS) for s in SEEDS}
    conditions = [("none", 0.0)] + [(d, a) for d in DIRECTIONS for a in ANGLES]
    rows = []
    for direction, angle in conditions:
        M = np.eye(len(ch)) if direction == "none" else displacement_matrix(ch, direction, angle)
        X = data["X"] if direction == "none" else displaced_inputs(M, subjects)
        mm = displacement_mm(ch, "right", 0.0) if direction == "none" else displacement_mm(ch, direction, angle)
        for (m, f, s), model in models.items():
            te = fold_of == f
            p = predict_proba(model, X[te])
            pred = (p >= 0.5).astype(int)
            for sj in np.unique(subj[te]):
                k = subj[te] == sj
                rows.append({"model": m, "fold": f, "seed": s, "subject": int(sj),
                             "direction": direction, "angle_deg": angle,
                             "shift_mm_Cz": mm["Cz"], "shift_mm_C3": mm["C3"],
                             "acc": float((pred[k] == y[te][k]).mean()),
                             "frac_pred_right": float(pred[k].mean())})
        print(f"B2.3 {direction} {angle:4.1f} deg (Cz {mm['Cz']:.1f} mm) done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "b2_3_displacement.csv", index=False, float_format="%.4f")
    return df


# ---------------------------------------------------------------------------
# Which layer is responsible?
# ---------------------------------------------------------------------------
def first_layer_sensitivity(out_dir: Path, seed: int) -> pd.DataFrame:
    """How much does the displacement change what each model's FIRST spatial
    weights see? A model applies spatial weights w to the data; with displaced
    data X' = M X, the effective weights on the true field are M^T w. The
    relative change ||M^T w - w|| / ||w|| measures how much a given
    misplacement distorts that model's spatial filters. Smooth (broad) filters
    change little; sharp, Laplacian-like filters change a lot.
    For channel-token models the per-channel signals are mixed instead:
    their sensitivity is the change of the input itself, ||M - I||."""
    data = load_net_data(out_dir / "audit.csv")
    ch = data["ch_names"]
    rows = []
    for direction in DIRECTIONS:
        for angle in ANGLES:
            M = displacement_matrix(ch, direction, angle)
            for m in ALL_MODELS:
                for f in range(N_FOLDS):
                    for s in SEEDS:
                        model = load_trained(f"main|{m}|f{f}|nall|s{s}", m)
                        if m in ("EEGNet", "TF-time"):
                            layer = model.conv_spatial if m == "EEGNet" else model.tok.conv_spatial
                            W = layer.weight.detach().numpy().reshape(layer.weight.shape[0], -1)   # (maps, 64)
                            # CAR is applied after displacement: fold it into the weights
                            Wc = W - W.mean(1, keepdims=True)
                            change = np.linalg.norm(Wc @ M - Wc, axis=1) / np.linalg.norm(Wc, axis=1)
                            # spatial roughness: energy of the weights after a
                            # Laplacian (neighbour difference) relative to the weights
                            rows.append({"model": m, "fold": f, "seed": s, "direction": direction,
                                         "angle_deg": angle, "filter_change": float(np.median(change))})
                        else:
                            change = np.linalg.norm(M - np.eye(len(ch)), axis=1).mean()
                            rows.append({"model": m, "fold": f, "seed": s, "direction": direction,
                                         "angle_deg": angle, "filter_change": float(change)})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "b2_3_first_layer_change.csv", index=False, float_format="%.4f")
    return df


@torch.no_grad()
def layer_similarity(out_dir: Path, seed: int, direction="right", angle=10.0) -> pd.DataFrame:
    """Clean-vs-displaced activation correlation at each stage of each model,
    on its own test subjects: where along the network does the displacement
    effect appear, and where is it amplified?"""
    data = load_net_data(out_dir / "audit.csv")
    subjects = kept_subjects(out_dir / "audit.csv")
    folds = make_folds(np.unique(data["subject"]), seed)
    fold_of = np.array([folds[int(s)] for s in data["subject"]])
    M = displacement_matrix(data["ch_names"], direction, angle)
    Xd_all = displaced_inputs(M, subjects)
    rows = []
    for m in ALL_MODELS:
        for f in range(N_FOLDS):
            for s in SEEDS:
                model = load_trained(f"main|{m}|f{f}|nall|s{s}", m)
                te = fold_of == f
                acts = {}
                for tag, X in (("clean", data["X"][te]), ("disp", Xd_all[te])):
                    acts[tag] = _stage_outputs(model, torch.from_numpy(X))
                for stage in acts["clean"]:
                    a, b = acts["clean"][stage].flatten(1), acts["disp"][stage].flatten(1)
                    # relative change of each trial's representation
                    rel = ((a - b).norm(dim=1) / a.norm(dim=1).clamp_min(1e-9)).median().item()
                    rows.append({"model": m, "fold": f, "seed": s, "stage": stage, "rel_change": rel})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "b2_3_layer_change.csv", index=False, float_format="%.4f")
    return df


def _stage_outputs(model, x):
    import torch.nn.functional as F
    out = {"input": x}
    if model.__class__.__name__ == "EEGNet":
        h = model.bn1(model.conv_time(x.unsqueeze(1)))
        out["temporal_conv"] = h
        h = F.elu(model.bn2(model.conv_spatial(h)))
        out["spatial_conv"] = h
        h = model.pool1(h)
        h = F.elu(model.bn3(model.conv_sep_point(model.conv_sep_depth(h))))
        out["separable_conv"] = h
        out["logits"] = model.classifier(model.pool2(h).flatten(1))
        return out
    tok = model.tokens(x)
    out["tokens"] = tok
    z = torch.cat([model.cls.expand(tok.shape[0], -1, -1), tok], dim=1)
    for i, b in enumerate(model.blocks):
        z = b(z)
        out[f"block{i + 1}"] = z
    out["logits"] = model.head(model.ln(z[:, 0]))
    return out


def displacement_figure(out_dir: Path):
    style()
    df = pd.read_csv(out_dir / "b2_3_displacement.csv")
    base = df[df.direction == "none"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
    for ax, (title, dirs) in zip(axes, (("lateral (cap slides to an ear)", ["left", "right"]),
                                        ("anterior–posterior", ["anterior", "posterior"]))):
        for m in ALL_MODELS:
            # per seed: mean over subjects; then mean and range over seeds
            def curve(direction):
                pts = []
                for a in [0.0] + ANGLES:
                    sub = base if a == 0 else df[(df.direction == direction) & (df.angle_deg == a)]
                    per_seed = sub[sub.model == m].groupby("seed").acc.mean()
                    pts.append(per_seed.values)
                return np.array(pts)
            both = np.mean([curve(d) for d in dirs], axis=0)       # average the two signs
            x = [0.0] + ANGLES
            ax.fill_between(x, both.min(1), both.max(1), color=MODEL_COLORS[m], alpha=0.18, lw=0)
            ax.plot(x, both.mean(1), color=MODEL_COLORS[m], marker="o", ms=4, label=m)
        ax.axhline(0.5, color=C["ink2"], lw=0.8, ls=":")
        ax.set_title(title, loc="left")
        ax.set_xlabel("cap rotation (deg); 10° ≈ 16 mm at Cz")
        # Models retrained without the ocular route (B2.5), lateral only.
        f = out_dir / "b2_3_displacement_no_ocular.csv"
        if f.exists() and dirs[0] == "left":
            no = pd.read_csv(f)
            for m in no.model.unique():
                pts = []
                for a in [0.0] + ANGLES:
                    sub = no[(no.model == m) & ((no.direction == "none") if a == 0 else
                                                (no.direction.isin(dirs) & (no.angle_deg == a)))]
                    pts.append(sub.groupby("seed").acc.mean().values)
                pts = np.array(pts)
                ax.plot([0.0] + ANGLES, pts.mean(1), color=MODEL_COLORS[m], ls="--", marker="s", ms=3,
                        label=f"{m}, ocular route removed")
    axes[0].set_ylabel("test accuracy (mean of 37 subjects)")
    axes[0].legend(loc="center right", fontsize=7)
    axes[1].legend(*axes[1].get_legend_handles_labels(), loc="center right", fontsize=7)
    fig.suptitle("B2.3 test-time electrode displacement (band = range over 3 seeds; both signs averaged)",
                 y=1.02)
    fig.savefig(out_dir / "figures" / "b2_3_displacement.png")
    plt.close(fig)


def run_displacement_no_ocular(out_dir: Path, seed: int, models=("EEGNet", "TF-time")) -> pd.DataFrame:
    """Same lateral displacement, applied to the models retrained WITHOUT the
    ocular route (B2.5). If the full models are robust because they read a
    broad ocular field, the motor-only models, which must read a focal
    sensorimotor pattern, should degrade more."""
    from task1.train import preprocess_variant
    data = load_net_data(out_dir / "audit.csv", "no_ocular")
    subjects = kept_subjects(out_dir / "audit.csv")
    folds = make_folds(np.unique(data["subject"]), seed)
    ch = list(data["ch_names"]); y, subj = data["y"], data["subject"]
    fold_of = np.array([folds[int(s)] for s in subj])
    rows = []
    for direction, angle in [("none", 0.0)] + [(d, a) for d in ("left", "right") for a in ANGLES]:
        if direction == "none":
            X = data["X"]
        else:
            M = displacement_matrix(ch, direction, angle)
            Xs = []
            for s in subjects:
                d = load_subject(s)
                Xd = np.einsum("ij,njt->nit", M, d["X"])
                Xs.append(preprocess_variant(Xd, d["times"], ch, "no_ocular"))
            X = np.concatenate(Xs)
        for m in models:
            for f in range(N_FOLDS):
                for sd in SEEDS:
                    model = load_trained(f"no_ocular|{m}|f{f}|nall|s{sd}", m)
                    te = fold_of == f
                    pred = (predict_proba(model, X[te]) >= 0.5).astype(int)
                    for sj in np.unique(subj[te]):
                        k = subj[te] == sj
                        rows.append({"model": m, "fold": f, "seed": sd, "subject": int(sj),
                                     "direction": direction, "angle_deg": angle,
                                     "acc": float((pred[k] == y[te][k]).mean())})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "b2_3_displacement_no_ocular.csv", index=False, float_format="%.4f")
    return df


def pattern_change(out_dir: Path) -> pd.DataFrame:
    """Signal-side explanation, independent of any model: how much does a
    cap rotation change the left-minus-right TOPOGRAPHY of each route?
      ocular  CAR, < 4 Hz amplitude at 0.25-0.6 s (the saccade, A3 R1)
      mu      CAR, 8-13 Hz log power at 0.5-4 s (motor ERD, A2)
    Patterns are averaged over subjects; the change is ||M p - p|| / ||p||.
    A spatially broad pattern changes little under a small rotation; a
    focal one (opposite-sign peaks a few cm apart) changes a lot."""
    from task1.a3_confounds import _filt
    subjects = kept_subjects(out_dir / "audit.csv")
    pats = {"ocular": [], "mu": []}
    for s in subjects:
        d = load_subject(s); t = d["times"]; y = d["y"]
        X = d["X"] - d["X"].mean(1, keepdims=True)
        lo = _filt(X, None, 4.0)
        lo = lo - lo[..., (t >= -0.5) & (t < 0)].mean(-1, keepdims=True)
        w = (t >= 0.25) & (t <= 0.6)
        pats["ocular"].append(lo[y == 0][..., w].mean((0, 2)) - lo[y == 1][..., w].mean((0, 2)))
        mu = np.log((_filt(X, 8, 13)[..., (t >= 0.5)] ** 2).mean(-1))
        pats["mu"].append(mu[y == 0].mean(0) - mu[y == 1].mean(0))
    ch = load_subject(subjects[0])["ch_names"]
    rows = []
    for name, P in pats.items():
        p = np.mean(P, axis=0); p = p - p.mean()
        for direction in DIRECTIONS:
            for angle in ANGLES:
                M = displacement_matrix(ch, direction, angle)
                q = M @ p; q = q - q.mean()
                rows.append({"pattern": name, "direction": direction, "angle_deg": angle,
                             "rel_change": float(np.linalg.norm(q - p) / np.linalg.norm(p)),
                             "corr_with_original": float(np.corrcoef(p, q)[0, 1])})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "b2_3_pattern_change.csv", index=False, float_format="%.4f")
    return df
