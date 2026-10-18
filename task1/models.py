"""B1 models, implemented from torch building blocks only.

EEGNet (Lawhern et al. 2018, arXiv:1611.08024), adapted from 128 Hz to this dataset's 160 Hz,
and a small transformer with three interchangeable tokenisers:
  time      time-patch tokens: all channels, one 250 ms segment per token
  chan_id   channel tokens: one electrode's time series per token + learned electrode identity
  chan_noid channel tokens without identity embedding
"""
from __future__ import annotations

import torch
import torch.nn as nn

N_CH, N_T, SFREQ = 64, 640, 160   # 64 electrodes, 0-4 s after the cue at 160 Hz


class TemporalConv(nn.Module):
    """EEGNet's first layer, Conv2d(1, F1, (1, k), padding='same', bias=False), computed as a
    Conv1d over every electrode's series. Same weights and output; ~2x faster on CPU."""

    def __init__(self, F1, k):
        super().__init__()
        self.conv = nn.Conv1d(1, F1, k, bias=False)
        self.pad = ((k - 1) // 2, k // 2)      # 'same' for an even kernel, as in Keras

    def forward(self, x):                      # (B, C, T) -> (B, F1, C, T)
        B, C, T = x.shape
        z = self.conv(nn.functional.pad(x.reshape(B * C, 1, T), self.pad))
        return z.view(B, C, -1, T).permute(0, 2, 1, 3)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EEGNet(nn.Module):
    """EEGNet-8,2. Hyperparameters changed for 160 Hz (paper values are for 128 Hz):
      kernel 1 (temporal):   64 -> 80 samples. The paper uses fs/2 = 0.5 s so the filter can
                             resolve down to 2 Hz; 0.5 s at 160 Hz is 80 samples.
      separable kernel:      16 -> 20 samples. After the first 4x pooling the rate is fs/4;
                             16 samples = 500 ms at 32 Hz, so 500 ms at 40 Hz is 20.
    Unchanged: F1=8, D=2, F2=16, pooling 4 then 8, ELU, max-norm 1 (depthwise spatial) and
    0.25 (dense), dropout 0.25 (the paper's cross-subject setting; 0.5 was within-subject).
    """

    def __init__(self, n_ch=N_CH, n_t=N_T, F1=8, D=2, F2=16, k1=80, k2=20, p=0.25, n_cls=2):
        super().__init__()
        self.block1 = nn.Sequential(TemporalConv(F1, k1), nn.BatchNorm2d(F1))
        # Depthwise spatial filter: one set of 64 electrode weights per temporal filter. This
        # is where EEGNet encodes electrode identity: weight [f, 0, c, 0] belongs to channel c.
        self.depthwise = nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False)
        self.block2 = nn.Sequential(
            nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(p),
            nn.Conv2d(F1 * D, F1 * D, (1, k2), padding="same", groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(p))
        self.classify = nn.Linear(F2 * (n_t // 32), n_cls)

    def forward(self, x):                      # x: (B, C, T)
        z = self.block2(self.depthwise(self.block1(x)))
        return self.classify(z.flatten(1))

    @torch.no_grad()
    def apply_max_norm(self):
        # Paper: max-norm constraint on the spatial and dense weights (renorm per output unit).
        self.depthwise.weight.data = torch.renorm(self.depthwise.weight.data, 2, 0, 1.0)
        self.classify.weight.data = torch.renorm(self.classify.weight.data, 2, 0, 0.25)


class TimePatchTokens(nn.Module):
    """Tokens = 16 time segments of 40 samples (250 ms), each summarising all channels.
    A full linear patch embedding (64 ch x 40 samples -> d) alone would cost ~41k
    parameters, 14x EEGNet; so it is factorised: a learned spatial projection to 8 virtual
    channels, then 2 temporal filters per virtual channel, ELU, and averaging within each
    patch. A learned positional embedding tells the encoder which segment is which."""

    def __init__(self, d=16, n_virtual=8, k=41, patch=40):
        super().__init__()
        self.spatial = nn.Linear(N_CH, n_virtual)
        self.temporal = nn.Sequential(
            nn.Conv1d(n_virtual, d, k, padding=k // 2, groups=n_virtual),
            nn.BatchNorm1d(d), nn.ELU(), nn.AvgPool1d(patch))
        self.pos = nn.Parameter(torch.zeros(1, N_T // patch, d))
        nn.init.normal_(self.pos, std=0.02)
        self.n_tokens = N_T // patch

    def forward(self, x):                      # (B, C, T) -> (B, 16, d)
        v = self.spatial(x.transpose(1, 2)).transpose(1, 2)
        return self.temporal(v).transpose(1, 2) + self.pos


class ChannelTokens(nn.Module):
    """Tokens = the 64 electrodes. Every channel's time series goes through the same small
    temporal encoder (8 filters, ELU, 8 time windows of 0.5 s, linear to d), so the token
    content carries no information about *which* electrode it came from. With identity=True
    a learned per-electrode embedding is added; without it, attention + mean pooling make
    the whole model invariant to permuting the electrodes."""

    def __init__(self, d=16, n_filt=8, k=41, pool=80, identity=True):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Conv1d(1, n_filt, k, padding=k // 2), nn.BatchNorm1d(n_filt), nn.ELU(),
            nn.AvgPool1d(pool), nn.Flatten(), nn.Linear(n_filt * (N_T // pool), d))
        self.identity = nn.Parameter(torch.zeros(1, N_CH, d)) if identity else None
        if identity:
            nn.init.normal_(self.identity, std=0.02)
        self.n_tokens = N_CH

    def forward(self, x):                      # (B, C, T) -> (B, 64, d)
        B, C, T = x.shape
        z = self.encode(x.reshape(B * C, 1, T)).view(B, C, -1)
        return z + self.identity if self.identity is not None else z


class EEGTransformer(nn.Module):
    """Tokeniser -> 2 pre-norm transformer layers (d=16, 2 heads) -> mean over tokens -> linear.
    Mean pooling (no CLS token) keeps the no-identity variant exactly permutation-invariant."""

    def __init__(self, tokeniser="time", d=16, n_layers=2, n_heads=2, ff=32, p=0.25, n_cls=2):
        super().__init__()
        self.tokeniser_name = tokeniser
        self.tokens = {"time": lambda: TimePatchTokens(d),
                       "chan_id": lambda: ChannelTokens(d, identity=True),
                       "chan_noid": lambda: ChannelTokens(d, identity=False)}[tokeniser]()
        layer = nn.TransformerEncoderLayer(d, n_heads, ff, dropout=p, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(p), nn.Linear(d, n_cls))

    def forward(self, x):
        return self.head(self.encoder(self.tokens(x)).mean(1))


def build(model: str, tokeniser: str | None = None) -> nn.Module:
    return EEGNet() if model == "eegnet" else EEGTransformer(tokeniser)


if __name__ == "__main__":
    e = n_params(EEGNet())
    for tok in ["time", "chan_id", "chan_noid"]:
        m = n_params(EEGTransformer(tok))
        print(f"{tok:10s} {m:5d} params = {m / e:.2f}x EEGNet ({e})")
