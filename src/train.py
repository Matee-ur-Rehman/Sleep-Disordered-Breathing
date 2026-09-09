"""
Training script for SPRNet on Sleep-EDF sleep-stage classification.

Uses the paper's fully-specified training protocol (Sec 4.3.3, Sec 4.4):
  - Adam optimizer, lr=1e-3, weight_decay=1e-4          [PAPER]
  - Cosine annealing LR schedule                         [PAPER]
  - Batch size 64                                        [PAPER]
  - Up to 100 epochs, early stopping patience 10 on val loss  [PAPER]
  - Repeated over 5 seeds, mean +/- std reported          [PAPER, Sec 4.3.3]

NOT yet implemented here (deliberately, per the phased plan):
  - IDSSA (transition priors, multi-objective loss) - single-task stage-only
    training for now, to validate the base SPRNet pipeline first.
  - Respiratory event / disorder loss - no labels on Sleep-EDF; disorder_head
    exists in the model but receives no gradient in this script.
  - The 5-seed repeat protocol as a full outer loop - this script runs ONE
    seed at a time (controlled by --seed) so we can first confirm a single
    run behaves sensibly (loss decreasing, no NaNs, reasonable Sleep-EDF
    accuracy) before spending CPU time on 5 full runs. A separate wrapper
    script can call this 5x with different seeds once we trust a single run.

CPU-first design note: this script runs correctly on CPU (uses whatever
device is available), which is deliberate per our workflow - verify
correctness here on CPU with a SMALL subset first (via --limit_batches or
--epochs), THEN move to Kaggle GPU for full-scale, full-epoch runs.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import load_manifest, build_subject_splits, SPRNetSequenceDataset, sprnet_collate_fn
from model import SPRNet, temporal_consistency_loss
from config import sprnet_cfg, train_cfg, seq_cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_metrics(all_preds, all_labels, num_classes=5):
    """
    Macro-averaged accuracy/precision/recall/F1 over the 5 sleep stages.
    [ASSUMED] The paper reports Accuracy/Precision/Recall/F1 but never states
    the averaging scheme for the multi-class task (macro vs weighted vs
    micro) - macro is the standard choice for imbalanced multi-class sleep
    staging (treats each class equally regardless of frequency, which is
    usually what's wanted given how skewed sleep-stage distributions are),
    and is what we use here. If we later find evidence the paper meant
    weighted or micro averaging, this is the one place to change.
    """
    all_preds = np.asarray(all_preds)
    all_labels = np.asarray(all_labels)

    accuracy = (all_preds == all_labels).mean()

    precisions, recalls, f1s = [], [], []
    for c in range(num_classes):
        tp = np.sum((all_preds == c) & (all_labels == c))
        fp = np.sum((all_preds == c) & (all_labels != c))
        fn = np.sum((all_preds != c) & (all_labels == c))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "accuracy": float(accuracy),
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1": float(np.mean(f1s)),
    }


def run_epoch(model, loader, optimizer, device, train=True):
    model.train() if train else model.eval()

    total_loss = 0.0
    n_batches = 0
    all_preds, all_labels = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for x_batch, y_batch in loader:
            x_batch = {m: t.to(device) for m, t in x_batch.items()}
            y_batch = y_batch.to(device)

            if train:
                optimizer.zero_grad()

            stage_logits, disorder_logits, h_prime = model(x_batch)

            # (batch, T, C) -> (batch, C, T) for CrossEntropyLoss
            stage_loss = torch.nn.functional.cross_entropy(
                stage_logits.permute(0, 2, 1), y_batch)
            temp_loss = temporal_consistency_loss(h_prime)

            # [ASSUMED] task_loss_lambda (Eq 21) applies to disorder_loss,
            # which we don't have labels for on Sleep-EDF, so effectively
            # only stage_loss + a small temporal-consistency term are active
            # here. IDSSA's full multi-objective loss (Eq 26) is not wired
            # in yet - this is single-task SPRNet training only, by design,
            # to validate the base pipeline first.
            loss = stage_loss + 0.1 * temp_loss  # [ASSUMED] 0.1 weight on temp_loss,
                                                    # not specified anywhere in the
                                                    # paper for this auxiliary term.

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            preds = stage_logits.argmax(dim=-1)  # (batch, T)
            all_preds.extend(preds.reshape(-1).cpu().numpy().tolist())
            all_labels.extend(y_batch.reshape(-1).cpu().numpy().tolist())

    avg_loss = total_loss / max(n_batches, 1)
    metrics = compute_metrics(all_preds, all_labels, num_classes=sprnet_cfg.num_stages)
    metrics["loss"] = avg_loss
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=train_cfg.seed)
    parser.add_argument("--epochs", type=int, default=train_cfg.epochs,
                         help="Max epochs. Use a small number (e.g. 2-3) for a "
                              "quick CPU sanity run before the full 100-epoch run.")
    parser.add_argument("--batch_size", type=int, default=train_cfg.batch_size)
    parser.add_argument("--lr", type=float, default=train_cfg.lr)
    parser.add_argument("--weight_decay", type=float, default=train_cfg.weight_decay)
    parser.add_argument("--patience", type=int, default=train_cfg.early_stopping_patience)
    parser.add_argument("--limit_batches", type=int, default=None,
                         help="If set, only run this many batches per epoch "
                              "(train and val) - for a fast CPU smoke test on "
                              "a fraction of the data, not real training.")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"\nLoading manifest and building splits (seed={args.seed})...")
    shard_files, channel_order = load_manifest(args.shard_dir)
    splits, split_subjects = build_subject_splits(shard_files)
    print(f"  train: {len(split_subjects['train'])} subjects, "
          f"val: {len(split_subjects['val'])} subjects, "
          f"test: {len(split_subjects['test'])} subjects")

    train_ds = SPRNetSequenceDataset(args.shard_dir, splits["train"], channel_order)
    val_ds = SPRNetSequenceDataset(args.shard_dir, splits["val"], channel_order)
    print(f"  train sequences: {len(train_ds)}, val sequences: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=sprnet_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=sprnet_collate_fn)

    if args.limit_batches:
        # Simple truncation wrapper for fast smoke tests - not used for real runs.
        from itertools import islice

        class LimitedLoader:
            def __init__(self, loader, n):
                self.loader, self.n = loader, n

            def __iter__(self):
                return islice(iter(self.loader), self.n)

        train_loader = LimitedLoader(train_loader, args.limit_batches)
        val_loader = LimitedLoader(val_loader, args.limit_batches)

    modality_counts = train_ds.modality_channel_counts()
    model = SPRNet(modality_channel_counts=modality_counts, cfg=sprnet_cfg, use_ecg=False)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = []

    print(f"\nStarting training: max {args.epochs} epochs, "
          f"patience {args.patience}, batch_size {args.batch_size}, lr {args.lr}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, train=False)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d}/{args.epochs} ({elapsed:.1f}s) | "
              f"train_loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.4f} | "
              f"val_loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f} "
              f"f1={val_metrics['f1']:.4f}")

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_dir / f"best_seed{args.seed}.pt")
            print(f"  -> new best val_loss, checkpoint saved.")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"\nEarly stopping: no val_loss improvement for "
                      f"{args.patience} epochs.")
                break

    history_path = checkpoint_dir / f"history_seed{args.seed}.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. History saved to {history_path}")
    print(f"Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()