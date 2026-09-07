"""
Run this LOCALLY (needs torch + your real preprocessed shards) to verify the
dataset loader end to end: splitting, windowing, tensor shapes, and that it
plugs into SPRNet correctly.

Usage: python scripts/test_dataset.py --shard_dir data/processed/sleep_edf_sc_full
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from torch.utils.data import DataLoader

from dataset import (
    load_manifest, build_subject_splits, SPRNetSequenceDataset, sprnet_collate_fn,
)
from model import SPRNet, temporal_consistency_loss
from config import sprnet_cfg, seq_cfg, SLEEP_EDF_CHANNELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    print(f"Loading manifest from {args.shard_dir}...")
    shard_files, channel_order = load_manifest(args.shard_dir)
    print(f"Found {len(shard_files)} shards, channels: {channel_order}")

    print("\nBuilding subject-wise splits...")
    splits, split_subjects = build_subject_splits(shard_files)
    for name in ["train", "val", "test"]:
        print(f"  {name}: {len(split_subjects[name])} subjects, {len(splits[name])} shards")

    # Sanity: no subject overlap (re-verify here too, not just in the earlier
    # numpy-only test, since this is the actual code path used for real training)
    train_s, val_s, test_s = set(split_subjects["train"]), set(split_subjects["val"]), set(split_subjects["test"])
    assert not (train_s & val_s) and not (train_s & test_s) and not (val_s & test_s), \
        "SUBJECT LEAKAGE DETECTED - do not proceed with training until this is fixed."
    print("  No subject leakage across splits - confirmed.")

    print(f"\nBuilding datasets (sequence_length={seq_cfg.sequence_length})...")
    train_ds = SPRNetSequenceDataset(args.shard_dir, splits["train"], channel_order)
    val_ds = SPRNetSequenceDataset(args.shard_dir, splits["val"], channel_order)
    test_ds = SPRNetSequenceDataset(args.shard_dir, splits["test"], channel_order)
    print(f"  train: {len(train_ds)} sequences, val: {len(val_ds)}, test: {len(test_ds)}")

    modality_counts = train_ds.modality_channel_counts()
    print(f"  modality channel counts: {modality_counts}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=sprnet_collate_fn)

    print("\nPulling one real batch...")
    x_batch, y_batch = next(iter(train_loader))
    for modality, tensor in x_batch.items():
        print(f"  x['{modality}'] shape: {tuple(tensor.shape)}")
    print(f"  y shape: {tuple(y_batch.shape)} (batch, sequence_length)")
    print(f"  y sample values (first sequence): {y_batch[0].tolist()}")

    expected_shape_prefix = (args.batch_size, seq_cfg.sequence_length)
    for modality, tensor in x_batch.items():
        assert tensor.shape[:2] == expected_shape_prefix, \
            f"{modality} shape mismatch: {tensor.shape}"
    assert y_batch.shape == expected_shape_prefix

    print("\nRunning this real batch through SPRNet end to end...")
    model = SPRNet(modality_channel_counts=modality_counts, cfg=sprnet_cfg, use_ecg=False)
    model.train()
    stage_logits, disorder_logits, h_prime = model(x_batch)
    print(f"  stage_logits shape: {tuple(stage_logits.shape)}")

    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(stage_logits.permute(0, 2, 1), y_batch)
    temp_loss = temporal_consistency_loss(h_prime)
    total_loss = loss + 0.1 * temp_loss
    print(f"  stage_loss: {loss.item():.4f}, temporal_consistency_loss: {temp_loss.item():.4f}")

    total_loss.backward()
    print("  backward() succeeded, gradients computed.")

    print("\nALL CHECKS PASSED - dataset loader is correctly wired to SPRNet.")


if __name__ == "__main__":
    main()