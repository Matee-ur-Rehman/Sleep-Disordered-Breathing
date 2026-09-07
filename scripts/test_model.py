"""
Run this LOCALLY (you have torch installed; this sandbox doesn't) to verify
SPRNet's forward pass, loss computation, and gradient flow before we trust
it on real data. This is a structural/plumbing test with synthetic random
data - it does NOT test whether the model learns anything meaningful, only
that shapes are consistent end-to-end and gradients actually flow.

Usage: python scripts/test_model.py
Expected: prints shape checks and "ALL CHECKS PASSED" with no errors.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from model import SPRNet, temporal_consistency_loss
from config import sprnet_cfg, build_channel_to_modality_map, SLEEP_EDF_CHANNELS, EPOCH_SEC

RESAMPLE_HZ = 100  # matches preprocessing.py's RESAMPLE_HZ


def main():
    torch.manual_seed(0)

    # --- Simulate Sleep-EDF's actual channel layout ---
    channel_order = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal",
                      "EMG submental", "Resp oro-nasal"]
    modality_idx = build_channel_to_modality_map(channel_order, SLEEP_EDF_CHANNELS)
    modality_channel_counts = {m: len(idxs) for m, idxs in modality_idx.items()}
    print("Modality -> channel indices:", modality_idx)
    print("Modality -> channel counts:", modality_channel_counts)

    batch_size = 4
    T = 10  # sequence length: 10 consecutive 30s epochs
    n_samples = EPOCH_SEC * RESAMPLE_HZ  # 3000

    # Build a fake batch: one big (batch, T, total_channels, n_samples) tensor,
    # then split into per-modality tensors using modality_idx - mimics what
    # the real dataset loader will need to do.
    total_channels = len(channel_order)
    full_batch = torch.randn(batch_size, T, total_channels, n_samples)

    x = {}
    for modality, idxs in modality_idx.items():
        x[modality] = full_batch[:, :, idxs, :]
        print(f"  x['{modality}'] shape: {tuple(x[modality].shape)}")

    # --- Build model (no ECG, matching Sleep-EDF) ---
    model = SPRNet(modality_channel_counts=modality_channel_counts,
                    cfg=sprnet_cfg, use_ecg=False)
    model.train()

    # --- Forward pass ---
    stage_logits, disorder_logits, h_prime = model(x)

    print(f"\nstage_logits shape: {tuple(stage_logits.shape)} "
          f"(expected: ({batch_size}, {T}, {sprnet_cfg.num_stages}))")
    assert stage_logits.shape == (batch_size, T, sprnet_cfg.num_stages)

    print(f"disorder_logits shape: {tuple(disorder_logits.shape)} "
          f"(expected: ({batch_size}, {T}, 1))")
    assert disorder_logits.shape == (batch_size, T, 1)

    print(f"h_prime shape: {tuple(h_prime.shape)} "
          f"(expected: ({batch_size}, {T}, {sprnet_cfg.attention_embed_dim}))")
    assert h_prime.shape == (batch_size, T, sprnet_cfg.attention_embed_dim)

    # --- Loss computation (stage classification only, as on Sleep-EDF) ---
    fake_stage_labels = torch.randint(0, sprnet_cfg.num_stages, (batch_size, T))
    stage_loss_fn = torch.nn.CrossEntropyLoss()
    # CrossEntropyLoss expects (N, C, ...) so permute: (batch, T, C) -> (batch, C, T)
    stage_loss = stage_loss_fn(stage_logits.permute(0, 2, 1), fake_stage_labels)
    print(f"\nstage_loss: {stage_loss.item():.4f} (should be a finite positive number)")
    assert torch.isfinite(stage_loss)

    temp_loss = temporal_consistency_loss(h_prime)
    print(f"temporal_consistency_loss: {temp_loss.item():.4f} (should be finite, >= 0)")
    assert torch.isfinite(temp_loss) and temp_loss.item() >= 0

    # Also exercise the disorder head with a dummy loss, purely to confirm
    # ITS wiring is correct too (gradient reaches it) - even though on real
    # Sleep-EDF training we would NOT include this term, since there are no
    # respiratory event labels for this dataset. This smoke test wants to
    # verify the whole architecture, not just the Sleep-EDF-relevant subset.
    fake_disorder_labels = torch.randint(0, 2, (batch_size, T, 1)).float()
    disorder_loss_fn = torch.nn.BCEWithLogitsLoss()
    disorder_loss = disorder_loss_fn(disorder_logits, fake_disorder_labels)
    print(f"disorder_loss (smoke-test only, not used in real Sleep-EDF training): "
          f"{disorder_loss.item():.4f}")
    assert torch.isfinite(disorder_loss)

    total_loss = stage_loss + 0.1 * temp_loss + disorder_loss

    # --- Gradient flow check ---
    model.zero_grad()
    total_loss.backward()

    n_params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    n_params_total = sum(1 for p in model.parameters())
    print(f"\nParameters with gradients: {n_params_with_grad}/{n_params_total}")
    assert n_params_with_grad == n_params_total, (
        "Some parameters received no gradient at all - likely a disconnected "
        "branch in the computation graph. Investigate before trusting this model."
    )

    # Check for NaN/Inf in any gradient
    bad_grads = [name for name, p in model.named_parameters()
                 if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad_grads, f"NaN/Inf gradients found in: {bad_grads}"

    n_total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal trainable parameters: {n_total_params:,}")
    print("(Paper reports 4.63M for the full model in Table 5 - ours will "
          "differ since that number covers the full SHHS 5-modality config "
          "with ECG, and includes IDSSA's contribution, which we haven't "
          "wired in yet. Not expected to match at this stage.)")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()