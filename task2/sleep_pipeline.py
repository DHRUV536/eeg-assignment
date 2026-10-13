"""
Sleep staging: CNN epoch-encoder + Transformer sequence model
==============================================================

Data   : Sleep-EDF Expanded, Sleep Cassette (SC) subset, fetched with MNE.
Task   : 5-class AASM staging (W, N1, N2, N3, REM). The model sees a window of
         SEQ_LEN consecutive 30-s epochs and predicts the stage of the CENTRE
         epoch of the window.
Metrics: accuracy, macro-F1 and Cohen's kappa on held-out data.

Run    : python sleep_pipeline.py
"""
import random

import mne
import numpy as np
import torch
import torch.nn as nn
from mne.datasets.sleep_physionet.age import fetch_data
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score)

SEED = 42
SUBJECTS = list(range(15))          # subjects 0-14, both nights where available
SEQ_LEN = 11                        # 11 x 30 s = 5.5 min of context
EPOCH_SEC = 30
CHANNELS = ["Fpz-Cz", "Pz-Oz", "horizontal"]   # names after infer_types=True
N_CLASSES = 5
CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]

# R&K -> AASM: stages 3 and 4 are merged into N3
STAGE_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}

REJECT_PTP = 500      # uV; epochs with larger peak-to-peak are electrode pops
BATCH = 32
TRAIN_EPOCHS = 12
STEPS_PER_EPOCH = 150
LR = 1e-3
D_MODEL = 64


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_files(subjects):
    return fetch_data(subjects=subjects, recording=[1, 2], on_missing="warn")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_recording(psg_file, hyp_file):
    raw = mne.io.read_raw_edf(psg_file, stim_channel="Event marker",
                              infer_types=True, preload=True, verbose="error")
    raw.pick(CHANNELS)
    raw.filter(0.3, 35.0, verbose="error")
    sf = int(raw.info["sfreq"])
    n_ep = raw.n_times // (EPOCH_SEC * sf)

    data = raw.get_data()[:, : n_ep * EPOCH_SEC * sf]
    X = data.reshape(len(CHANNELS), n_ep, EPOCH_SEC * sf).transpose(1, 0, 2)

    # one label per 30-s epoch from the hypnogram
    annot = mne.read_annotations(hyp_file)
    y = np.zeros(n_ep, dtype=np.int64)
    for onset, dur, desc in zip(annot.onset, annot.duration, annot.description):
        start, stop = int(onset // EPOCH_SEC), int((onset + dur) // EPOCH_SEC)
        y[start:stop] = STAGE_MAP.get(desc, 0)

    # drop electrode pops
    keep = np.ptp(X, axis=-1).max(axis=1) < REJECT_PTP
    X, y = X[keep], y[keep]

    # standardise each epoch so the network is insensitive to amplitude drift
    X = (X - X.mean(axis=-1, keepdims=True)) / (X.std(axis=-1, keepdims=True) + 1e-8)
    return X.astype(np.float32), y


class SeqDataset(torch.utils.data.Dataset):
    """Windows of SEQ_LEN consecutive epochs from each recording."""

    def __init__(self, recordings):
        self.recs = recordings
        self.index = [(r, i) for r, (X, y) in enumerate(recordings)
                      for i in range(len(y) - SEQ_LEN + 1)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        r, i = self.index[k]
        X, y = self.recs[r]
        return torch.from_numpy(X[i:i + SEQ_LEN]), int(y[i + SEQ_LEN - 1])


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class EpochEncoder(nn.Module):
    def __init__(self, n_ch, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_ch, 16, 50, stride=6), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(8),
            nn.Conv1d(16, 32, 8), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, d, 8), nn.BatchNorm1d(d), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):                      # (N, C, T) -> (N, d)
        return self.net(x).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=64):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-np.log(10000.0) / d))
        pe = torch.zeros(max_len, d)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, z):                      # (B, L, d)
        return z + self.pe[: z.size(1)]


class SleepTransformer(nn.Module):
    def __init__(self, n_ch, d=D_MODEL, n_classes=N_CLASSES):
        super().__init__()
        self.encoder = EpochEncoder(n_ch, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=4, dim_feedforward=128, dropout=0.1)
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.pos = PositionalEncoding(d)
        self.head = nn.Linear(d, n_classes)

    def forward(self, x):                      # x: (B, L, C, T)
        B, L = x.shape[:2]
        z = self.encoder(x.flatten(0, 1)).view(B, L, -1)
        z = self.transformer(z)
        z = self.pos(z)
        return self.head(z[:, L // 2])


# --------------------------------------------------------------------------
# Train / evaluate
# --------------------------------------------------------------------------
@torch.no_grad()
def predict(model, ds):
    model.eval()
    loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False)
    ys, ps = [], []
    for xb, yb in loader:
        ps.append(model(xb).argmax(1).numpy())
        ys.append(yb.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def main():
    set_seed(SEED)
    files = get_files(SUBJECTS)
    random.Random(SEED).shuffle(files)
    n_test = max(1, int(0.2 * len(files)))
    test_files, train_files = files[:n_test], files[n_test:]
    print(f"train recordings: {len(train_files)}  test recordings: {len(test_files)}")

    train_ds = SeqDataset([load_recording(*f) for f in train_files])
    test_ds = SeqDataset([load_recording(*f) for f in test_files])

    y_train = np.array([train_ds.recs[r][1][i + SEQ_LEN - 1] for r, i in train_ds.index])
    counts = np.bincount(y_train, minlength=N_CLASSES)
    print("train label counts:", dict(zip(CLASS_NAMES, counts.tolist())))
    weights = torch.tensor(len(y_train) / (N_CLASSES * np.maximum(counts, 1)), dtype=torch.float32)

    model = SleepTransformer(n_ch=len(CHANNELS))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    sampler = torch.utils.data.RandomSampler(train_ds, replacement=True,
                                             num_samples=BATCH * STEPS_PER_EPOCH)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH, sampler=sampler)

    best_f1, best_state = -1.0, None
    for ep in range(TRAIN_EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        y_true, y_pred = predict(model, test_ds)
        f1 = f1_score(y_true, y_pred, average="macro")
        print(f"epoch {ep:2d}  loss {loss.item():.3f}  macro-F1 {f1:.3f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    y_true, y_pred = predict(model, test_ds)
    print("\n=== held-out results ===")
    print(f"accuracy {accuracy_score(y_true, y_pred):.3f}")
    print(f"macro-F1 {f1_score(y_true, y_pred, average='macro'):.3f}")
    print(f"kappa    {cohen_kappa_score(y_true, y_pred):.3f}")
    print("per-class F1:", dict(zip(CLASS_NAMES, np.round(
        f1_score(y_true, y_pred, average=None, labels=range(N_CLASSES)), 3).tolist())))
    print(confusion_matrix(y_true, y_pred, labels=range(N_CLASSES)))


if __name__ == "__main__":
    main()
