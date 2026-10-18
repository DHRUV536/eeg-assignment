"""B2.3: what does the best transformer (and, for comparison, EEGNet) rely on?

Specs fixed in PREDICTIONS.md P4 before this ran:
  Integrated Gradients (zero baseline = training mean after z-scoring, 32 steps, predicted
  class) on every test trial; token deletion (top-k attributed vs random time tokens);
  attribution per channel group and time; channel-group deletion; per-test-subject Spearman
  correlations with A2 mu LI and the A3 decoders. Also, for write-up (a), the A3 route decoders
  trained cross-subject on the same split.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from task1.a3_confounds import cross_subject_accuracy
from task1.train import MODEL_DIR, OUT, SEED, load_trained, predict_proba, prepare, subject_split

BEST_TF = ("tf-time_nall_stopacc", "transformer", "time")   # highest val acc (D28)
EEGNET = ("eegnet_nall", "eegnet", None)
IG_STEPS = 32
PATCH = 40                                                   # time-token length (samples)
K_TOKENS = list(range(0, 9))
N_RANDOM = 10
GROUP_ORDER = ["frontal/ocular", "sensorimotor", "parieto-occipital", "temporal"]


def channel_groups(ch_names) -> dict[str, list[int]]:
    g = {k: [] for k in GROUP_ORDER}
    for i, c in enumerate(ch_names):
        if c.startswith(("FC", "C")):                # FC*, C*, CP*
            g["sensorimotor"].append(i)
        elif c.startswith(("Fp", "AF", "F")) and not c.startswith("FT"):
            g["frontal/ocular"].append(i)
        elif c.startswith(("P", "O", "Iz")):         # P*, PO*, O*, Iz
            g["parieto-occipital"].append(i)
        else:                                        # FT*, T*, TP*
            g["temporal"].append(i)
    assert sum(map(len, g.values())) == len(ch_names)
    return g


def integrated_gradients(model, X: np.ndarray, batch: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """IG w.r.t. the predicted class, zero baseline; midpoint Riemann sum with IG_STEPS.
    Returns (attributions with X's shape, predicted classes)."""
    model.eval()
    pred = predict_proba(model, X).argmax(1)
    out = np.zeros_like(X)
    alphas = (np.arange(IG_STEPS) + 0.5) / IG_STEPS
    for i in range(0, len(X), batch):
        x = torch.from_numpy(X[i:i + batch])
        tgt = torch.from_numpy(pred[i:i + batch])
        g = torch.zeros_like(x)
        for a in alphas:
            xa = (a * x).requires_grad_(True)
            logit = model(xa).gather(1, tgt[:, None]).sum()
            g += torch.autograd.grad(logit, xa)[0]
        out[i:i + batch] = (x * g / IG_STEPS).numpy()
    return out, pred


def attributions(run, D) -> dict:
    """IG for one model on the test set, cached (EEGNet IG takes ~1-2 min on CPU)."""
    path = MODEL_DIR / f"{run[0]}_ig.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    model = load_trained(*run)
    ig, pred = integrated_gradients(model, D["Xte"])
    a = np.abs(ig)
    res = dict(per_channel=a.sum(2),                                   # (trial, 64)
               per_token=a.reshape(len(a), a.shape[1], -1, PATCH).sum((1, 3)),  # (trial, 16)
               pred=pred)
    tmp = MODEL_DIR / f"tmp_{os.getpid()}_{path.name}"
    np.savez(tmp, **res)
    os.replace(tmp, path)
    return res


def eval_acc(model, X, y) -> tuple[float, float]:
    p = predict_proba(model, X)
    return float((p.argmax(1) == y).mean()), float(p[np.arange(len(y)), y].mean())


def token_deletion(model, D, per_token) -> pd.DataFrame:
    """Replace the input of the k most-attributed 250 ms tokens (per trial) with the training
    mean (0 after z-scoring) vs k random tokens."""
    X, y = D["Xte"], D["yte"]
    order = np.argsort(-per_token, axis=1)
    rng = np.random.default_rng(SEED)
    rows = []
    for k in K_TOKENS:
        Xd = X.copy()
        for n in range(len(X)):
            for t in order[n, :k]:
                Xd[n, :, t * PATCH:(t + 1) * PATCH] = 0
        acc, pt = eval_acc(model, Xd, y)
        rows.append(dict(k=k, kind="most-attributed", draw=0, acc=acc, p_true=pt))
        for r in range(N_RANDOM if k else 1):
            Xr = X.copy()
            for n in range(len(X)):
                for t in rng.choice(per_token.shape[1], k, replace=False):
                    Xr[n, :, t * PATCH:(t + 1) * PATCH] = 0
            acc, pt = eval_acc(model, Xr, y)
            rows.append(dict(k=k, kind="random", draw=r, acc=acc, p_true=pt))
    return pd.DataFrame(rows)


def group_deletion(model, D, groups) -> pd.DataFrame:
    X, y = D["Xte"], D["yte"]
    rows = [dict(group="none", draw=0, n_channels=0, acc=eval_acc(model, X, y)[0])]
    for g, idx in groups.items():
        Xd = X.copy()
        Xd[:, idx] = 0
        rows.append(dict(group=g, draw=0, n_channels=len(idx), acc=eval_acc(model, Xd, y)[0]))
    rng = np.random.default_rng(SEED + 1)
    for r in range(2 * N_RANDOM):
        idx = rng.choice(X.shape[1], 17, replace=False)
        Xd = X.copy()
        Xd[:, idx] = 0
        rows.append(dict(group="random 17", draw=r, n_channels=17, acc=eval_acc(model, Xd, y)[0]))
    return pd.DataFrame(rows)


def heog_regression(D, pairs=(("F7", "F8"), ("AF7", "AF8"), ("Fp1", "Fp2"))):
    """Post-hoc (added after P4's results, D30). Zeroing one region cannot remove a signal that
    volume-conducts to frontal, temporal and (via the reference) all channels. Instead regress
    the horizontal-EOG proxies out of every channel, Gratton-style: one coefficient per
    channel and proxy, least squares over all *training-subject* samples, applied unchanged to
    the test subjects. Returns the cleaned test data."""
    ch = list(D["ch_names"])
    ia, ib = [ch.index(a) for a, _ in pairs], [ch.index(b) for _, b in pairs]
    R_tr = D["Xtr"][:, ia] - D["Xtr"][:, ib]                    # (trial, 3, time)
    A = R_tr.transpose(0, 2, 1).reshape(-1, len(pairs))          # (samples, 3)
    Y = D["Xtr"].transpose(0, 2, 1).reshape(-1, D["Xtr"].shape[1])
    B, *_ = np.linalg.lstsq(A, Y, rcond=None)                    # (3, 64)
    R_te = D["Xte"][:, ia] - D["Xte"][:, ib]
    return (D["Xte"] - np.einsum("nkt,kc->nct", R_te, B)).astype(np.float32)


def main(out_dir: Path = OUT) -> None:
    split = subject_split()
    D = prepare(split["train"], split)
    ch = list(D["ch_names"])
    groups = channel_groups(ch)
    models = {"transformer": BEST_TF, "EEGNet": EEGNET}

    share_rows, gdel, attr_ch = [], [], {}
    Xclean = heog_regression(D)
    for name, run in models.items():
        model = load_trained(*run)
        A = attributions(run, D)
        tot = A["per_channel"].sum(1, keepdims=True)
        frac = A["per_channel"] / tot                             # per trial, sums to 1
        attr_ch[name] = frac.mean(0)
        for g in GROUP_ORDER:
            idx = groups[g]
            share_rows.append(dict(model=name, group=g, n_channels=len(idx),
                                   share=float(frac[:, idx].sum(1).mean()),
                                   share_per_channel_x64=float(frac[:, idx].sum(1).mean() / len(idx) * 64)))
        tf_share = A["per_token"] / A["per_token"].sum(1, keepdims=True)
        for t in range(tf_share.shape[1]):
            share_rows.append(dict(model=name, group=f"time {t * 0.25:.2f}-{(t + 1) * 0.25:.2f}s",
                                   n_channels=64, share=float(tf_share[:, t].mean()),
                                   share_per_channel_x64=np.nan))
        gd = group_deletion(model, D, groups)
        gd = pd.concat([gd, pd.DataFrame([dict(group="HEOG regressed (post-hoc)", draw=0, n_channels=64,
                                               acc=eval_acc(model, Xclean, D["yte"])[0])])])
        gd.insert(0, "model", name)
        gdel.append(gd)
        if name == "transformer":
            tdel = token_deletion(model, D, A["per_token"])
            tdel.to_csv(out_dir / "b2_3_token_deletion.csv", index=False)
            tok_share = tf_share.mean(0)
        else:
            eeg_tok_share = tf_share.mean(0)
        print(f"[B2.3] {name} done", flush=True)

    shares = pd.DataFrame(share_rows)
    shares.round(4).to_csv(out_dir / "b2_3_attribution_shares.csv", index=False)
    gdel = pd.concat(gdel)
    gdel.round(4).to_csv(out_dir / "b2_3_group_deletion.csv", index=False)
    pd.DataFrame({"channel": ch, **{f"share_{k}": v for k, v in attr_ch.items()}}).round(5) \
        .to_csv(out_dir / "b2_3_channel_attribution.csv", index=False)

    # Per-test-subject correlations with Part A.
    runs = pd.read_csv(out_dir / "b2_runs.csv").set_index("run_id")
    table = pd.read_csv(out_dir / "per_subject_table.csv").set_index("subject")
    corr_rows, subj_rows = [], []
    for name, run in models.items():
        per = pd.Series(eval(runs.loc[run[0], "per_subject"])).rename(lambda s: int(s))
        for s, a in per.items():
            subj_rows.append(dict(model=name, subject=s, acc=a))
        for col in ["li_mu_db", "acc_ocular", "acc_visual_evoked", "acc_motor"]:
            r = spearmanr(per.values, table.loc[per.index, col].values)
            corr_rows.append(dict(model=name, part_a_measure=col, n_subjects=len(per),
                                  spearman_rho=r.statistic, p=r.pvalue))
    corr = pd.DataFrame(corr_rows)
    corr.round(4).to_csv(out_dir / "b2_3_correlations.csv", index=False)

    # Per-subject accuracy after HEOG regression (post-hoc, D30).
    post = []
    for name, run in models.items():
        p = predict_proba(load_trained(*run), Xclean).argmax(1)
        for subj in np.unique(D["ste"]):
            m = D["ste"] == subj
            post.append(dict(model=name, subject=int(subj), acc_heog_regressed=float((p[m] == D["yte"][m]).mean()),
                             li_mu_db=table.loc[subj, "li_mu_db"]))
    post = pd.DataFrame(post)
    post.round(4).to_csv(out_dir / "b2_3_heog_regressed_per_subject.csv", index=False)
    for name, g in post.groupby("model"):
        r = spearmanr(g.acc_heog_regressed, g.li_mu_db)
        print(f"[B2.3 post-hoc] {name} after HEOG regression: acc {g.acc_heog_regressed.mean():.3f}; "
              f"rho with mu LI {r.statistic:+.2f} (p={r.pvalue:.2f})")

    # Write-up (a): the A3 routes as cross-subject decoders on the same split.
    xs = pd.concat([cross_subject_accuracy(r, split["train"], split["test"])
                    for r in ["ocular", "visual_evoked", "motor"]])
    xs.round(4).to_csv(out_dir / "b2_3_cross_subject_route_decoders.csv", index=False)

    plot(tdel, shares, gdel, corr, attr_ch, ch, tok_share, eeg_tok_share,
         pd.DataFrame(subj_rows), table, out_dir)

    t = tdel.groupby(["kind", "k"]).acc.mean().unstack(0)
    print("[B2.3] token deletion (acc):\n" + t.round(3).to_string())
    print("[B2.3] attribution share per group (x64/n = per-channel ratio to uniform):\n" +
          shares[shares.group.isin(GROUP_ORDER)].round(3).to_string(index=False))
    print("[B2.3] group deletion:\n" + gdel.groupby(["model", "group"]).acc.mean().round(3).to_string())
    print("[B2.3] correlations:\n" + corr.round(3).to_string(index=False))
    print("[B2.3] cross-subject route decoders (test subjects): " +
          ", ".join(f"{r} {g.acc.mean():.3f}" for r, g in xs.groupby("route")) +
          f"; pooled: " + ", ".join(f"{r} {(g.acc * g.n).sum() / g.n.sum():.3f}" for r, g in xs.groupby("route")))


def plot(tdel, shares, gdel, corr, attr_ch, ch, tok_share, eeg_tok_share, subj, table, out_dir):
    fig = plt.figure(figsize=(16, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)
    # (a) deletion curves
    ax = fig.add_subplot(gs[0, 0])
    top = tdel[tdel.kind == "most-attributed"].groupby("k").acc.mean()
    rnd = tdel[tdel.kind == "random"].groupby("k").acc
    ax.plot(top.index, top.values, "o-", color="tab:red", label="most-attributed tokens")
    ax.plot(rnd.mean().index, rnd.mean().values, "o-", color="tab:grey", label="random tokens (mean of 10)")
    ax.fill_between(rnd.mean().index, rnd.min().values, rnd.max().values, color="tab:grey", alpha=0.2)
    ax.axhline(0.5, color="k", lw=0.5)
    ax.set_xlabel("time tokens (250 ms) deleted per trial")
    ax.set_ylabel("test accuracy")
    ax.set_title("Transformer: IG deletion test", fontsize=9)
    ax.legend(fontsize=7)
    # (b) attribution over time
    ax = fig.add_subplot(gs[0, 1])
    tt = np.arange(len(tok_share)) * 0.25 + 0.125
    ax.bar(tt - 0.05, tok_share, width=0.1, label="transformer", color="tab:orange")
    ax.bar(tt + 0.05, eeg_tok_share, width=0.1, label="EEGNet", color="tab:blue")
    ax.axhline(1 / len(tok_share), color="k", lw=0.5, ls="--")
    ax.set_xlabel("time from cue (s)")
    ax.set_ylabel("share of |IG|")
    ax.set_title("Where in time the attribution lies (dashed = uniform)", fontsize=9)
    ax.legend(fontsize=7)
    # (c, d) topomaps
    info = mne.create_info(ch, 160.0, "eeg")
    info.set_montage("standard_1005")
    for j, name in enumerate(["transformer", "EEGNet"]):
        ax = fig.add_subplot(gs[0, 2 + j])
        mne.viz.plot_topomap(attr_ch[name] * 64, info, axes=ax, show=False, cmap="Reds",
                             vlim=(0, None), contours=0)
        ax.set_title(f"{name}: |IG| per channel (1 = uniform)", fontsize=9)
    # (e) group shares per channel
    ax = fig.add_subplot(gs[1, 0])
    s = shares[shares.group.isin(GROUP_ORDER)]
    for i, (name, c) in enumerate([("transformer", "tab:orange"), ("EEGNet", "tab:blue")]):
        v = s[s.model == name].set_index("group").loc[GROUP_ORDER].share_per_channel_x64
        ax.bar(np.arange(4) + (i - 0.5) * 0.38, v.values, width=0.38, color=c, label=name)
    ax.axhline(1, color="k", lw=0.5, ls="--")
    ax.set_xticks(range(4), GROUP_ORDER, fontsize=7)
    ax.set_ylabel("|IG| per channel / uniform")
    ax.set_title("Attribution per channel, by region", fontsize=9)
    ax.legend(fontsize=7)
    # (f) group deletion
    ax = fig.add_subplot(gs[1, 1])
    order = ["none"] + GROUP_ORDER + ["random 17", "HEOG regressed (post-hoc)"]
    for i, (name, c) in enumerate([("transformer", "tab:orange"), ("EEGNet", "tab:blue")]):
        g = gdel[gdel.model == name].groupby("group").acc
        ax.bar(np.arange(len(order)) + (i - 0.5) * 0.38, g.mean().loc[order].values, width=0.38,
               color=c, label=name, yerr=g.std().fillna(0).loc[order].values, capsize=2)
    ax.axhline(0.5, color="k", lw=0.5)
    ax.set_xticks(range(len(order)), ["none", "frontal/\nocular (17)", "sensori-\nmotor (21)",
                                      "parieto-\noccip. (18)", "temporal\n(8)", "random\n17 (×20)",
                                      "HEOG\nregressed*"], fontsize=7)
    ax.set_ylim(0.3, 0.9)
    ax.set_ylabel("test accuracy")
    ax.set_title("Zeroing channel groups at test time (*post-hoc: HEOG proxies regressed out)", fontsize=8)
    ax.legend(fontsize=7)
    # (g, h) per-subject accuracy vs A3 ocular and A2 LI
    for j, (col, lab) in enumerate([("acc_ocular", "A3 ocular decoder accuracy (within subject)"),
                                    ("li_mu_db", "A2 mu lateralisation index (dB)")]):
        ax = fig.add_subplot(gs[1, 2 + j])
        for name, c in [("transformer", "tab:orange"), ("EEGNet", "tab:blue")]:
            d = subj[subj.model == name]
            x = table.loc[d.subject, col].values
            r = corr[(corr.model == name) & (corr.part_a_measure == col)].iloc[0]
            ax.scatter(x, d.acc, color=c, label=f"{name}: ρ={r.spearman_rho:+.2f} (p={r.p:.2f})")
        ax.axhline(0.5, color="k", lw=0.5)
        ax.set_xlabel(lab, fontsize=8)
        ax.set_ylabel("model accuracy (test subject)")
        ax.set_title("Per test subject (n = 10)", fontsize=9)
        ax.legend(fontsize=7)
    fig.savefig(out_dir / "b2_3_reliance.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
