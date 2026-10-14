"""
B2.4 - What does the best transformer rely on?

Two token-importance scores per test trial, from the trained models of all
5 folds x 3 seeds (each evaluated on its own test subjects only):
  attention     CLS-row attention of the last block, averaged over heads
  grad x input  token-level gradient x input: for token t with embedding
                e_t, score_t = | sum_d  e_t,d * d(logit_pred - logit_other)/d e_t,d |
                (first-order estimate of how much the decision changes if
                the token is removed)
Faithfulness: delete the k highest-scoring tokens (attention ranking,
attribution ranking, or a random ranking) and measure test accuracy. A score
that reflects what the model uses should make accuracy fall faster than
random deletion. Tokens are deleted by dropping them from the sequence, so
the model never sees an out-of-distribution placeholder.

Mapping to Part A: input-level gradient x input, summed over time per
electrode (spatial map) and over electrodes per time bin (time course),
compared with the A3 single-channel decoding maps (ocular < 4 Hz amplitude
vs mu/beta power).
"""
from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch

from task1.plotting import C, plt, style
from task1.train import N_FOLDS, SEEDS, load_net_data, load_trained, make_folds
from task1.models import SFREQ

RANDOM_REPEATS = 5


def _scores(model, x):
    """Attention and grad x input token scores, and the predicted class."""
    tok = model.tokens(x).detach().requires_grad_(True)
    logits = model.encode(tok, keep_attn=True)
    pred = logits.argmax(1)
    other = 1 - pred
    margin = logits.gather(1, pred[:, None]) - logits.gather(1, other[:, None])
    margin.sum().backward()
    gxi = (tok.grad * tok).sum(-1).abs().detach()                    # (B, L)
    attn = model.blocks[-1].last_attn.mean(1)[:, 0, 1:].detach()     # CLS row, (B, L)
    return attn, gxi, pred.detach()


@torch.no_grad()
def _accuracy_after_deletion(model, x, y, order, k):
    """Drop the first k tokens of `order` (B, L) from each trial's sequence."""
    tok = model.tokens(x)
    if k > 0:
        keep = order[:, k:].sort(dim=1).values
        tok = torch.gather(tok, 1, keep[..., None].expand(-1, -1, tok.shape[-1]))
    pred = model.encode(tok).argmax(1)
    return (pred == y).float().mean().item()


def run_reliance(out_dir: Path, seed: int, model_name: str) -> pd.DataFrame:
    data = load_net_data(out_dir / "audit.csv")
    folds = make_folds(np.unique(data["subject"]), seed)
    fold_of = np.array([folds[int(s)] for s in data["subject"]])
    rng = torch.Generator().manual_seed(seed + 4000)
    rows, attn_all, gxi_all, sal_ch, sal_t = [], [], [], [], []
    for f in range(N_FOLDS):
        te = fold_of == f
        x = torch.from_numpy(data["X"][te]); y = torch.from_numpy(data["y"][te])
        for s in SEEDS:
            model = load_trained(f"main|{model_name}|f{f}|nall|s{s}", model_name)
            attn, gxi, pred = _scores(model, x)
            attn_all.append(attn.numpy()); gxi_all.append(gxi.numpy())
            L = attn.shape[1]
            ks = sorted({0, 1, 2, 3, 4, 6, 8} if L <= 16 else {0, 2, 4, 8, 16, 24, 32, 48})
            ks = [k for k in ks if k < L]
            orders = {"attention": attn.argsort(1, descending=True),
                      "grad_x_input": gxi.argsort(1, descending=True)}
            for name, order in orders.items():
                for k in ks:
                    rows.append({"fold": f, "seed": s, "ranking": name, "k": k, "frac": k / L,
                                 "acc": _accuracy_after_deletion(model, x, y, order, k)})
            for r in range(RANDOM_REPEATS):
                order = torch.stack([torch.randperm(L, generator=rng) for _ in range(len(y))])
                for k in ks:
                    rows.append({"fold": f, "seed": s, "ranking": "random", "k": k, "frac": k / L,
                                 "acc": _accuracy_after_deletion(model, x, y, order, k), "rep": r})
            ch_map, t_map = input_saliency(model, x)
            sal_ch.append(ch_map); sal_t.append(t_map)
        print(f"B2.4 {model_name} fold {f} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "b2_4_deletion.csv", index=False, float_format="%.4f")

    # Token importance profiles and the input-level maps
    attn_m = np.concatenate(attn_all).mean(0); gxi_m = np.concatenate(gxi_all)
    gxi_m = (gxi_m / gxi_m.sum(1, keepdims=True)).mean(0)
    prof = pd.DataFrame({"token": np.arange(len(attn_m)), "attention": attn_m, "grad_x_input": gxi_m})
    if model_name.startswith("TF-chan"):
        prof["channel"] = data["ch_names"]
    else:
        prof["t_start_s"] = prof.token * 24 / SFREQ
    prof.to_csv(out_dir / "b2_4_token_importance.csv", index=False, float_format="%.5f")
    sal = pd.DataFrame({"channel": data["ch_names"], "saliency": np.mean(sal_ch, axis=0)})
    sal.to_csv(out_dir / "b2_4_channel_saliency.csv", index=False, float_format="%.5f")
    np.save(out_dir.parents[1] / "cache" / "b2_4_time_saliency.npy", np.mean(sal_t, axis=0))
    return df


def input_saliency(model, x):
    """|gradient x input| at the network input, per channel and per sample."""
    x = x.clone().requires_grad_(True)
    logits = model(x)
    pred = logits.argmax(1)
    margin = logits.gather(1, pred[:, None]) - logits.gather(1, (1 - pred)[:, None])
    margin.sum().backward()
    g = (x.grad * x).abs().detach()                                   # (B, C, T)
    g = g / g.sum(dim=(1, 2), keepdim=True)
    return g.sum(2).mean(0).numpy(), g.sum(1).mean(0).numpy()


def map_to_part_a(out_dir: Path, model_names) -> pd.DataFrame:
    """Spearman correlation of each model's channel saliency with the A3
    single-channel decoding maps."""
    from scipy.stats import spearmanr
    topo = pd.read_csv(out_dir / "a3_channel_topography.csv", index_col=0)
    rows = []
    for m in model_names:
        f = out_dir / f"b2_4_channel_saliency_{m}.csv"
        if not f.exists():
            continue
        sal = pd.read_csv(f).set_index("channel").saliency.reindex(topo.index)
        for col in topo.columns:
            r = spearmanr(sal, topo[col])
            rows.append({"model": m, "A3_map": col, "spearman_rho": r.statistic, "p": r.pvalue})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "b2_4_map_vs_part_a.csv", index=False, float_format="%.4g")
    return df


def reliance_figures(out_dir: Path, model_name: str):
    style()
    df = pd.read_csv(out_dir / "b2_4_deletion.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.3), gridspec_kw={"width_ratios": [1, 1, 1]})
    ax = axes[0]
    for name, col in (("attention", C["orange"]), ("grad_x_input", C["blue"]), ("random", C["grey"])):
        sub = df[df.ranking == name]
        per_seed = sub.groupby(["frac", "seed"]).acc.mean().unstack()          # mean over folds
        x = per_seed.index.values
        ax.fill_between(x, per_seed.min(1), per_seed.max(1), color=col, alpha=0.2, lw=0)
        ax.plot(x, per_seed.mean(1), color=col, marker="o", ms=3, label=name.replace("_", " × "))
    ax.axhline(0.5, color=C["ink2"], lw=0.8, ls=":")
    ax.set_xlabel("fraction of tokens deleted"); ax.set_ylabel("test accuracy")
    ax.set_title(f"Deletion test, {model_name}", loc="left"); ax.legend(fontsize=8)

    prof = pd.read_csv(out_dir / "b2_4_token_importance.csv")
    sal = pd.read_csv(out_dir / f"b2_4_channel_saliency_{model_name}.csv")
    info = mne.create_info(list(sal.channel), 160.0, "eeg"); info.set_montage("standard_1005")
    if "channel" in prof:
        im, _ = mne.viz.plot_topomap(prof.grad_x_input.values, info, axes=axes[1], show=False,
                                     cmap="Blues", contours=0)
        axes[1].set_title("token grad × input per electrode", loc="left")
    else:
        t = prof.t_start_s.values + 0.15
        axes[1].plot(t, prof.attention / prof.attention.sum(), color=C["orange"], marker="o", ms=3, label="attention")
        axes[1].plot(t, prof.grad_x_input, color=C["blue"], marker="o", ms=3, label="grad × input")
        axes[1].set_xlabel("token centre (s after cue)"); axes[1].set_ylabel("share of importance")
        axes[1].set_title("importance by time token", loc="left"); axes[1].legend(fontsize=8)
    im, _ = mne.viz.plot_topomap(sal.saliency.values, info, axes=axes[2], show=False, cmap="Blues", contours=0)
    axes[2].set_title("input grad × input per electrode", loc="left")
    fig.savefig(out_dir / "figures" / "b2_4_reliance.png")
    plt.close(fig)


def saliency_all_models(out_dir: Path, seed: int, model_names):
    """Input-level saliency maps for every model (for B2.3/B2.4 comparisons)."""
    data = load_net_data(out_dir / "audit.csv")
    folds = make_folds(np.unique(data["subject"]), seed)
    fold_of = np.array([folds[int(s)] for s in data["subject"]])
    for m in model_names:
        maps, tmaps = [], []
        for f in range(N_FOLDS):
            x = torch.from_numpy(data["X"][fold_of == f])
            for s in SEEDS:
                ch_map, t_map = input_saliency(load_trained(f"main|{m}|f{f}|nall|s{s}", m), x)
                maps.append(ch_map); tmaps.append(t_map)
        pd.DataFrame({"channel": data["ch_names"], "saliency": np.mean(maps, axis=0)}).to_csv(
            out_dir / f"b2_4_channel_saliency_{m}.csv", index=False, float_format="%.5f")
        np.save(out_dir.parents[1] / "cache" / f"b2_4_time_saliency_{m}.npy", np.mean(tmaps, axis=0))
