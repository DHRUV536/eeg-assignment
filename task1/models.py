"""
B1 - Models, implemented from PyTorch building blocks only.

EEGNet-8,2 (Lawhern et al. 2018, arXiv:1611.08024), adapted from 128 Hz to
160 Hz, and one transformer design with three tokenisations:
  "time"        time-patch tokens: all channels, one time segment per token
  "chan_id"     channel tokens: one electrode's time series per token, plus a
                learned electrode-identity embedding
  "chan_noid"   channel tokens without identity embedding
All three share the same encoder and head, so differences between them come
from the tokeniser.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_CH = 64
# Networks see the data resampled to 80 Hz (DECISIONS D20): all content of
# interest (mu/beta <= 30 Hz, EOG, visual evoked) is below the 40 Hz Nyquist,
# and it halves training cost for the ~200 training runs B2 needs.
SFREQ = 80
N_TIMES = 318                     # 0 to 3.975 s after the cue at 80 Hz


# ---------------------------------------------------------------------------
# EEGNet
# ---------------------------------------------------------------------------
class EEGNet(nn.Module):
    """EEGNet-8,2 with every 128 Hz-specific length rescaled, in seconds, to 80 Hz.

    The paper's lengths are in samples at 128 Hz. Changed (paper -> here):
      kernel_length 64 -> 40   the first temporal filter stays 0.5 s long, so it
                               still spans 4 cycles of 8 Hz mu and resolves
                               down to 2 Hz.
      pool1          4 -> 2    post-pool rate 40 Hz (paper 32 Hz); 32 Hz is not
                               reachable from 80 Hz with an integer pool, and
                               rounding up keeps more temporal detail.
      sep_kernel    16 -> 20   the separable filter stays 0.5 s long at the
                               post-pool rate (20 samples at 40 Hz).
    Unchanged: F1=8, D=2, F2=16, pool2 8 (final rate 5 Hz vs 4 Hz), dropout
    0.5 (the paper's cross-subject value), max-norm 1.0 on the depthwise
    spatial filter and 0.25 on the classifier, ELU, batch norm.
    """

    def __init__(self, n_ch=N_CH, n_times=N_TIMES, n_classes=2, F1=8, D=2, F2=16,
                 kernel_length=40, pool1=2, sep_kernel=20, pool2=8, dropout=0.5):
        super().__init__()
        self.conv_time = nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        # Depthwise spatial filter: one 64-weight spatial pattern per (temporal filter, D).
        # These are the weights B2.3 points to.
        self.conv_spatial = nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, pool1))
        self.drop1 = nn.Dropout(dropout)
        self.conv_sep_depth = nn.Conv2d(F1 * D, F1 * D, (1, sep_kernel), padding=(0, sep_kernel // 2),
                                        groups=F1 * D, bias=False)
        self.conv_sep_point = nn.Conv2d(F1 * D, F2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, pool2))
        self.drop2 = nn.Dropout(dropout)
        with torch.no_grad():
            n_flat = self._features(torch.zeros(1, n_ch, n_times)).shape[1]
        self.classifier = nn.Linear(n_flat, n_classes)

    def _features(self, x):
        x = x.unsqueeze(1)                                   # (B, 1, C, T)
        x = self.bn1(self.conv_time(x))
        x = F.elu(self.bn2(self.conv_spatial(x)))
        x = self.drop1(self.pool1(x))
        x = self.conv_sep_point(self.conv_sep_depth(x))
        x = self.drop2(self.pool2(F.elu(self.bn3(x))))
        return x.flatten(1)

    def forward(self, x):
        return self.classifier(self._features(x))

    def apply_max_norm(self):
        """Max-norm constraints from the paper, applied after each optimiser step."""
        with torch.no_grad():
            w = self.conv_spatial.weight
            w.copy_(torch.renorm(w, 2, 0, 1.0))
            w = self.classifier.weight
            w.copy_(torch.renorm(w, 2, 0, 0.25))


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------
class Block(nn.Module):
    """Pre-norm transformer block that can return its attention weights (B2.4).
    nn.TransformerEncoderLayer does not expose them, hence this small block
    built from nn.MultiheadAttention."""

    def __init__(self, d, n_heads, d_ff, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d))
        self.drop = nn.Dropout(dropout)
        self.last_attn = None

    def forward(self, z, keep_attn=False):
        h = self.ln1(z)
        a, w = self.attn(h, h, h, need_weights=keep_attn, average_attn_weights=False)
        if keep_attn:
            self.last_attn = w                               # (B, heads, L, L)
        z = z + self.drop(a)
        return z + self.drop(self.ff(self.ln2(z)))


def sinusoidal(n, d):
    pos = torch.arange(n).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class TimePatchTokenizer(nn.Module):
    """Tokens = time segments; each token sees all 64 channels.

    Temporal filter bank (same design as the channel tokeniser) -> per-filter
    spatial filters over all 64 electrodes (EEGNet-style depthwise) -> cut
    into 13 non-overlapping 0.3 s patches -> each patch average-pooled to 4
    values per map -> linear embedding per patch.
    """

    def __init__(self, d, n_times, n_filters=8, kernel=40, depth=2, patch_s=24, pool=6):
        super().__init__()
        self.conv_time = nn.Conv2d(1, n_filters, (1, kernel), padding=(0, kernel // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.conv_spatial = nn.Conv2d(n_filters, n_filters * depth, (N_CH, 1), groups=n_filters, bias=False)
        self.bn2 = nn.BatchNorm2d(n_filters * depth)
        self.patch_s = patch_s
        self.n_tokens = n_times // patch_s
        self.pool = nn.AvgPool1d(pool)                       # 53 samples -> 4 values per patch
        per_patch = n_filters * depth * (patch_s // pool)
        self.embed = nn.Linear(per_patch, d)
        self.register_buffer("pos", sinusoidal(self.n_tokens, d))

    def forward(self, x):                                     # (B, C, T)
        h = self.bn1(self.conv_time(x.unsqueeze(1)))
        h = F.elu(self.bn2(self.conv_spatial(h)))[:, :, 0, :]  # (B, maps, T)
        h = h[..., : self.n_tokens * self.patch_s]
        B, M, _ = h.shape
        h = h.reshape(B, M, self.n_tokens, self.patch_s).permute(0, 2, 1, 3)   # (B, L, M, S)
        h = self.pool(h.reshape(B * self.n_tokens, M, self.patch_s)).reshape(B, self.n_tokens, -1)
        return self.embed(h) + self.pos


class ChannelTokenizer(nn.Module):
    """Tokens = electrodes; each token is one channel's whole time series.

    The same temporal filter bank is applied to every channel (weights
    shared), pooled into 0.5 s segments and embedded. With use_id=False the
    tokens carry no information about WHICH electrode they come from, so the
    encoder + CLS readout is a function of the unordered set of channel
    signals (permutation-invariant).
    """

    def __init__(self, d, n_times, n_filters=8, kernel=40, n_seg=8, use_id=True):
        super().__init__()
        self.conv_time = nn.Conv1d(1, n_filters, kernel, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm1d(n_filters)
        # 8 segments of ~0.5 s each over the 4 s trial
        self.pool = nn.AdaptiveAvgPool1d(n_seg)
        self.embed = nn.Linear(n_filters * n_seg, d)
        self.use_id = use_id
        self.id_embed = nn.Parameter(torch.zeros(N_CH, d)) if use_id else None
        if use_id:
            nn.init.normal_(self.id_embed, std=0.02)

    def forward(self, x):                                     # (B, C, T)
        B, Cn, T = x.shape
        h = self.conv_time(x.reshape(B * Cn, 1, T))
        h = self.pool(F.elu(self.bn(h)))                      # (B*C, filters, segs)
        tok = self.embed(h.flatten(1)).reshape(B, Cn, -1)
        if self.use_id:
            tok = tok + self.id_embed
        return tok


class EEGTransformer(nn.Module):
    def __init__(self, tokenizer="time", n_times=N_TIMES, n_classes=2, d=16, n_heads=2, d_ff=32,
                 n_layers=2, dropout=0.25):
        super().__init__()
        self.kind = tokenizer
        if tokenizer == "time":
            self.tok = TimePatchTokenizer(d, n_times)
        elif tokenizer in ("chan_id", "chan_noid"):
            self.tok = ChannelTokenizer(d, n_times, use_id=(tokenizer == "chan_id"))
        else:
            raise ValueError(tokenizer)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.blocks = nn.ModuleList([Block(d, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_classes)
        self.drop = nn.Dropout(dropout)

    def tokens(self, x):
        return self.tok(x)

    def encode(self, tok, keep_attn=False):
        z = torch.cat([self.cls.expand(tok.shape[0], -1, -1), tok], dim=1)
        z = self.drop(z)
        for b in self.blocks:
            z = b(z, keep_attn=keep_attn)
        return self.head(self.ln(z[:, 0]))

    def forward(self, x, keep_attn=False):
        return self.encode(self.tokens(x), keep_attn=keep_attn)

    def apply_max_norm(self):
        pass


def build_model(name: str, n_times: int = N_TIMES) -> nn.Module:
    if name == "EEGNet":
        return EEGNet(n_times=n_times)
    return EEGTransformer({"TF-time": "time", "TF-chan-id": "chan_id",
                           "TF-chan-noid": "chan_noid"}[name], n_times=n_times)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
