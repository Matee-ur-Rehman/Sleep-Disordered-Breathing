"""
Dataset loading for SPRNet training on the sharded preprocessed output from
preprocessing.py.

Handles two things the paper specifies precisely, and one it doesn't:
  1. [PAPER, Sec 4.2.1] 8:1:1 subject-wise split - BOTH nights of the same
     subject must land in the same split, or we leak information (a model
     could learn subject-specific quirks from night 1 in train and be
     evaluated on night 2 of the SAME subject in test - not a fair test of
     generalization to unseen subjects). We split by subject ID, not by
     recording/shard.
  2. [PAPER] No subject appears in more than one subset - enforced directly
     by splitting subject IDs first, then assigning every shard belonging to
     that subject to the same split.
  3. [ASSUMED, NOT SPECIFIED] Sequence length/stride for grouping consecutive
     epochs into one BiLSTM training sequence - see config.SequenceConfig.

This module is split into PURE-PYTHON/NUMPY functions (subject parsing, split
assignment, window-index construction) that can be tested without torch, and
a torch.utils.data.Dataset class that depends on torch (tested separately,
locally, since this sandbox has no torch installed).
"""
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    train_cfg, seq_cfg, SLEEP_EDF_CHANNELS, build_channel_to_modality_map,
)

SHARD_NAME_RE = re.compile(r"^SC4(\d{2})(\d)")  # e.g. "SC4001" -> subject "00", night "1"


def parse_subject_id(shard_filename: str) -> str:
    """
    Extract the 2-digit subject ID from a Sleep-EDF SC shard filename, e.g.
    'SC4001.npz' -> '00', 'SC4011.npz' -> '01'. Both nights of the same
    subject share the same subject ID by construction of the naming scheme.
    """
    m = SHARD_NAME_RE.match(shard_filename)
    if not m:
        raise ValueError(f"Shard filename '{shard_filename}' doesn't match the "
                          f"expected Sleep-EDF SC naming pattern (SC4ssN...).")
    return m.group(1)


def load_manifest(shard_dir):
    shard_dir = Path(shard_dir)
    manifest_path = shard_dir / "manifest.npz"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.npz found at {manifest_path}. "
                                 f"Run preprocessing.py first.")
    manifest = np.load(manifest_path, allow_pickle=True)
    shard_files = [str(s) for s in manifest["shard_files"]]
    channels = [str(c) for c in manifest["channels"]]
    return shard_files, channels


def build_subject_splits(shard_files, train_frac=train_cfg.train_frac,
                          val_frac=train_cfg.val_frac, test_frac=train_cfg.test_frac,
                          seed=train_cfg.split_seed):
    """
    Group shard files by subject, then split SUBJECTS (not shards/recordings)
    into train/val/test according to the given fractions, with a fixed seed
    for reproducibility. Returns a dict: {'train': [shard_files...], 'val':
    [...], 'test': [...]}.
    """
    subject_to_shards = {}
    for f in shard_files:
        sid = parse_subject_id(f)
        subject_to_shards.setdefault(sid, []).append(f)

    subjects = sorted(subject_to_shards.keys())
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(subjects).tolist()

    n = len(shuffled)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    # Remainder goes to test, so fractions always sum to n exactly regardless
    # of rounding - avoids accidentally dropping a subject due to rounding.
    train_subjects = shuffled[:n_train]
    val_subjects = shuffled[n_train:n_train + n_val]
    test_subjects = shuffled[n_train + n_val:]

    def shards_for(subject_list):
        out = []
        for s in subject_list:
            out.extend(subject_to_shards[s])
        return sorted(out)

    splits = {
        "train": shards_for(train_subjects),
        "val": shards_for(val_subjects),
        "test": shards_for(test_subjects),
    }
    split_subjects = {
        "train": train_subjects, "val": val_subjects, "test": test_subjects,
    }
    return splits, split_subjects


def build_window_index(shard_dir, shard_files, sequence_length=seq_cfg.sequence_length,
                        sequence_stride=seq_cfg.sequence_stride):
    """
    For each shard, determine how many non-overlapping (or strided) sequence
    windows of `sequence_length` consecutive epochs it contains, WITHOUT
    loading the shard's full X array (only its epoch count, via a cheap
    lookup of y's length) - keeps this index-building step fast even across
    150+ shards.

    Returns a list of (shard_filename, start_epoch_idx) tuples - one entry
    per training sequence. Windows never cross a shard/recording boundary,
    since consecutive epochs from two different recordings aren't actually
    temporally adjacent.
    """
    shard_dir = Path(shard_dir)
    index = []
    for shard_file in shard_files:
        with np.load(shard_dir / shard_file, allow_pickle=True) as d:
            n_epochs = len(d["y"])
        start = 0
        while start + sequence_length <= n_epochs:
            index.append((shard_file, start))
            start += sequence_stride
    return index


# ---------------------------------------------------------------------------
# Everything below this point requires torch. Not testable in a torch-free
# sandbox - verify locally via scripts/test_dataset.py.
# ---------------------------------------------------------------------------
import torch
from torch.utils.data import Dataset


class ShardCache:
    """
    LRU cache so that consecutive window lookups from the same shard file
    don't repeatedly re-open and decompress the same .npz.

    [FIX, real performance issue found during Kaggle GPU testing] The
    original default (max_size=4) was far too small once combined with
    DataLoader shuffle=True: with 153 total shards and only 4 cached, a
    shuffled epoch touches many different shards per batch, causing constant
    cache eviction and re-decompression - a CPU-side I/O bottleneck that can
    dominate wall-clock time even though the model itself is small and the
    GPU is fast. Confirmed empirically: batches were taking 15-30s each,
    implausibly slow for a ~1M-parameter model on a T4, strongly suggesting
    I/O-bound rather than compute-bound behavior.

    Each shard is roughly 70 MB in memory once decompressed (185,641 total
    epochs x 5 channels x 3000 samples x 4 bytes / 153 shards). A cache of
    40 shards costs ~2.8 GB RAM, comfortably affordable, and covers roughly
    a quarter of the full dataset at once - a much more reasonable working
    set for shuffled access than 4 ever was.
    """

    def __init__(self, shard_dir, max_size=40):
        self.shard_dir = Path(shard_dir)
        self.max_size = max_size
        self._cache = {}   # shard_file -> (X, y)
        self._order = []   # LRU order, most-recently-used at the end

    def get(self, shard_file):
        if shard_file in self._cache:
            self._order.remove(shard_file)
            self._order.append(shard_file)
            return self._cache[shard_file]

        with np.load(self.shard_dir / shard_file, allow_pickle=True) as d:
            X, y = d["X"], d["y"]

        self._cache[shard_file] = (X, y)
        self._order.append(shard_file)
        if len(self._order) > self.max_size:
            oldest = self._order.pop(0)
            del self._cache[oldest]
        return X, y


class SPRNetSequenceDataset(Dataset):
    """
    Yields sequences of `sequence_length` consecutive 30s epochs, split into
    per-modality tensors ready for SPRNet.forward()'s dict input format.

    __getitem__ returns:
        x: dict {modality: tensor(sequence_length, n_channels_modality, n_samples)}
        y_stage: tensor(sequence_length,) of integer stage labels

    (No batch dim here - that's added by the DataLoader via collate_fn.)
    (No ECG/disorder labels - not available on Sleep-EDF. When we add SHHS
    support, this class will need a variant that also yields ecg_features
    and y_disorder; kept separate rather than overloading this class with
    None-handling branches everywhere, per the plan to build SHHS support
    once access is granted.)
    """

    def __init__(self, shard_dir, shard_files, channel_order,
                 channel_groups=SLEEP_EDF_CHANNELS,
                 sequence_length=seq_cfg.sequence_length,
                 sequence_stride=seq_cfg.sequence_stride,
                 cache_size=40):
        self.shard_dir = Path(shard_dir)
        self.channel_order = channel_order
        self.modality_indices = build_channel_to_modality_map(channel_order, channel_groups)
        self.sequence_length = sequence_length
        self.window_index = build_window_index(shard_dir, shard_files,
                                                 sequence_length, sequence_stride)
        self.cache = ShardCache(shard_dir, max_size=cache_size)

    def __len__(self):
        return len(self.window_index)

    def modality_channel_counts(self):
        return {m: len(idxs) for m, idxs in self.modality_indices.items()}

    def __getitem__(self, idx):
        shard_file, start = self.window_index[idx]
        X, y = self.cache.get(shard_file)  # X: (n_epochs, n_channels, n_samples), y: (n_epochs,)

        end = start + self.sequence_length
        X_seq = X[start:end]   # (T, n_channels, n_samples)
        y_seq = y[start:end]   # (T,)

        x_dict = {}
        for modality, idxs in self.modality_indices.items():
            x_dict[modality] = torch.from_numpy(X_seq[:, idxs, :].copy()).float()

        y_tensor = torch.from_numpy(y_seq.copy()).long()
        return x_dict, y_tensor


def sprnet_collate_fn(batch):
    """
    Custom collate: batch is a list of (x_dict, y_tensor) pairs. Stack each
    modality's tensors across the batch dimension separately, since the
    model expects a dict of (batch, T, n_channels, n_samples) tensors.
    """
    x_dicts, y_tensors = zip(*batch)

    modalities = x_dicts[0].keys()
    x_batched = {
        modality: torch.stack([xd[modality] for xd in x_dicts], dim=0)
        for modality in modalities
    }
    y_batched = torch.stack(y_tensors, dim=0)
    return x_batched, y_batched