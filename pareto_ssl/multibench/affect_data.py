"""Loaders for the MultiBench affect datasets (CMU-MOSI, CMU-MOSEI, UR-FUNNY, MUsTARD).

These read the standard MultiBench pickle directly, so the benchmark can load data
without a third-party checkout. The preprocessing follows the MultiBench reference
exactly, because the published numbers depend on it:

  * rows whose text features are all zero are dropped;
  * -inf in the audio channel becomes 0;
  * every modality is cut to start at the first non-zero text frame (align=True);
  * labels binarise at 0 for the sentiment datasets, while UR-FUNNY carries its own
    0/1 humour flag and is passed through unchanged;
  * sequences are padded per batch, not to a fixed length.

`tests_affect_equivalence.py` asserts tensor-level equality against the reference
loader, so a drift in any of those rules fails loudly instead of quietly moving a
published result.
"""
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

_HERE = Path(__file__).resolve().parent
_PROG = _HERE.parent.parent
_CATALOG = _HERE / "data_catalog.json"

AFFECT_DATASETS = ("mosi", "mosei", "humor", "sarcasm", "mustard")


def data_path(dataset):
    """Resolve a dataset to a file on disk.

    $MB_DATA_CATALOG wins, then data_catalog.json next to this file. Paths in the
    catalog may be relative to the repository root.
    """
    path = os.environ.get("MB_DATA_CATALOG") or _CATALOG
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no dataset catalog at {path}. Copy data_catalog.example.json to "
            f"data_catalog.json and point each entry at your copy of the data.")
    with open(path) as f:
        catalog = json.load(f)
    if dataset not in catalog:
        raise KeyError(f"{dataset!r} is not in {path}; known: {sorted(catalog)}")
    p = catalog[dataset]["path"]
    return p if os.path.isabs(p) else str(_PROG / p)


def data_root(dataset):
    """Directory for a non-pickle dataset (avmnist, enrico), same catalog."""
    return data_path(dataset)


def _drop_textless(split):
    """MultiBench drops samples whose text features are entirely zero."""
    keep = [i for i, t in enumerate(split["text"]) if np.asarray(t).sum() != 0]
    return {k: np.asarray(v)[keep] for k, v in split.items()}


class AffectDataset(Dataset):
    """One split of a MultiBench affect dataset, returning (modalities, label)."""

    SPLIT_ALIAS = {"val": "valid", "valid": "valid", "train": "train", "test": "test"}

    def __init__(self, data_path, dataset, split="train",
                 modalities=("vision", "text"), task="classification", align=True):
        self.name = dataset
        self.modalities = tuple(modalities)
        self.task = task
        self.align = align

        with open(data_path, "rb") as f:
            raw = pickle.load(f)
        data = _drop_textless(raw[self.SPLIT_ALIAS[split]])
        audio = np.asarray(data["audio"], dtype=np.float64)
        audio[audio == -np.inf] = 0.0
        data["audio"] = audio
        self.data = data

    def __len__(self):
        return len(self.data["vision"])

    def _label(self, i):
        flag = self.data["labels"][i]
        if self.task != "classification":
            return flag
        # UR-FUNNY ships a 0/1 humour flag; the sentiment sets carry a real score.
        return [flag] if self.name == "humor" else ([[1]] if flag > 0 else [[0]])

    def __getitem__(self, i):
        chans = {m: self.data[m][i] for m in ("vision", "audio", "text")}
        if self.align:
            start = chans["text"].nonzero()[0][0]
            chans = {m: v[start:].astype(np.float32) for m, v in chans.items()}
        else:
            chans = {m: v[v.nonzero()[0][0]:].astype(np.float32)
                     for m, v in chans.items()}
        return [chans[m] for m in self.modalities], self._label(i)


class Augment:
    """MultiBench 'drop+noise': each applied to all modalities together, p=0.5."""

    def __init__(self, spec="drop+noise", p=0.5):
        self.ops = [s for s in spec.split("+") if s]
        self.p = p

    @staticmethod
    def _drop(x, frac=(0.0, 0.8)):
        idx = np.random.choice(
            len(x[0]), round(np.random.uniform(*frac) * len(x[0])), replace=False)
        out = []
        for xi in x:
            xi = np.copy(xi)
            xi[idx] = 0.0
            out.append(xi)
        return out

    @staticmethod
    def _noise(x, std=0.1):
        return [xi + np.random.randn(*xi.shape).astype(np.float32) * std for xi in x]

    def __call__(self, x):
        for op in self.ops:
            if np.random.rand() >= self.p:
                continue
            x = self._drop(x) if op == "drop" else self._noise(x)
        return x


class AffectSSLDataset(AffectDataset):
    """Returns two independently augmented views of the same sample."""

    def __init__(self, *a, augmentations="drop+noise", **kw):
        super().__init__(*a, **kw)
        self.augment = Augment(augmentations)

    def __getitem__(self, i):
        x, _ = super().__getitem__(i)
        return self.augment(x), self.augment(x)


def collate_timeseries(batch, max_seq_length=None):
    """Pad a batch of variable-length sequences, one list entry per modality."""
    out = []
    for m in range(len(batch[0])):
        padded = pad_sequence([torch.as_tensor(x[m]) for x in batch], batch_first=True)
        if max_seq_length is not None and max_seq_length > padded.shape[1]:
            pad = torch.zeros(padded.shape[0], max_seq_length - padded.shape[1],
                              padded.shape[2], dtype=padded.dtype)
            padded = torch.cat([padded, pad], dim=1)
        out.append(padded)
    return out


def collate_affect(batch, max_seq_length=None):
    labels = np.array([y for _, y in batch]).reshape(len(batch))
    return collate_timeseries([x for x, _ in batch], max_seq_length), torch.tensor(labels)


def collate_affect_ssl(batch, max_seq_length=None):
    return (collate_timeseries([a for a, _ in batch], max_seq_length),
            collate_timeseries([b for _, b in batch], max_seq_length))
