# SDB / SPRNet+IDSSA Reproduction — Status

Reproducing: Zhang et al., "Deep Learning for Multimodal Physiological Signal
Based Assessment of Sleep Disordered Breathing," Sci Rep (2026),
https://doi.org/10.1038/s41598-026-56111-6

## Where we are

- [x] Project scaffold, config with all hyperparameters tagged [PAPER]/[ASSUMED]/[DERIVED]
- [x] Sleep-EDF (Sleep Cassette) download script (MNE-based, verified subject indexing: 78 subjects, indices 0-82 excluding {39,68,69,78,79})
- [x] Preprocessing pipeline: load, clean, z-score normalize, epoch, outlier-exclude
- [x] Unit-tested the array-processing logic (normalization, outlier exclusion, noise injection) on synthetic data at realistic scale
- [ ] NOT yet run against real downloaded data (requires your machine's internet access — this sandbox has none)
- [ ] SPRNet model (Conv branches + BiLSTM + attention + dual heads) — not started
- [ ] IDSSA (transition priors + multi-objective loss) — not started
- [ ] Baselines (SVM/CNN/LSTM/ResNet/Transformer/MobileNetV2) — not started
- [ ] SHHS pipeline — blocked on NSRR approval

## How to run what exists so far (on your machine, in VS Code)

```bash
cd sdb_reproduction
pip install -r requirements.txt

# 1. Download (quick test first: 3 subjects, ~few hundred MB)
python scripts/download_sleep_edf.py --out ./data/sleep-edf-sc --subjects 0-2

# 2. Preprocess (matches the --limit to what you downloaded)
python src/preprocessing.py --data_dir ./data/sleep-edf-sc --out ./data/processed/test_run.npz --limit 6

# Once that works end to end, scale up:
python scripts/download_sleep_edf.py --out ./data/sleep-edf-sc --subjects all
python src/preprocessing.py --data_dir ./data/sleep-edf-sc --out ./data/processed/sleep_edf_sc.npz
```

The preprocessing script prints per-recording epoch counts and exclusions as
it runs — please share that output with me once you run it on real data, so
we can sanity-check things like: are exclusion rates reasonable (should be a
small fraction, not a large chunk)? Do all recordings actually have the
airflow channel, or only some? Does the final stage distribution roughly
match Table 1 of the paper (W: 30.1%, N1: 8.9%, N2: 38.4%, N3: 12.5%, REM:
10.1%)?

## Full list of unresolved ambiguities so far (carried from earlier discussion, plus new ones found while coding)

1. λ in Eq. 21 (task loss weight) — assumed 1.0
2. η (transition matrix adaptation rate, Eq. 23) — assumed 0.1, weakly inferred from Sec 4.4's mention of a "smoothing factor of 0.1" (not explicitly linked to η)
3. λ in Eq. 24 (transition regularization strength, distinct symbol reuse) — assumed 0.1
4. β in Eq. 26 (uncertainty loss weight) — assumed 0.1
5. τ_st (per-stage persistence timescale, Eq. 25) — assumed a single shared 120s value, not per-stage
6. τ (attention sliding context window, Eq. 15) — assumed ±3 steps
7. Fusion mechanism after modality-specific extraction — assumed simple concatenation (paper's Fig. 1 box has no equations)
8. R-peak detection algorithm for ECG HR/HRV features — not yet chosen (blocked on SHHS access anyway)
9. Exact SVM handcrafted feature set (baseline) — not yet chosen
10. Baseline architecture specifics (CNN/LSTM/ResNet/Transformer/MobileNetV2 depth/width adapted to 1D) — not yet chosen
11. Precision/Recall/F1 averaging scheme (macro/micro/weighted) for the 5-class task — assumed macro (most common for imbalanced multi-class sleep staging), not yet implemented/confirmed
12. Negative-sample downsampling ratio for SHHS respiratory task — not yet chosen (blocked on SHHS access)
13. How overlapping 30s/15s-stride windows reconcile with one-label-per-30s-epoch evaluation — **resolved by using non-overlapping epochs as the main path** (config.USE_OVERLAP=False); paper's literal 15s-stride text is not followed in the default pipeline, flagged clearly in config.py
14. **NEW**: outlier-exclusion granularity (per-sample vs. per-epoch statistic) — paper's literal per-sample reading is mathematically unworkable (excludes almost all data); resolved via per-epoch RMS z-score against the recording's own epoch-RMS distribution — see detailed comment in `src/preprocessing.py::build_epoch_array`
15. **NEW**: common resampling rate for multimodal alignment — paper doesn't state one; assumed 100Hz (matches Sleep-EDF's fastest native channels)
16. **NEW**: whether Gaussian noise injection (Eq., Sec 4.2.2) applies to train only or all splits — assumed train-only, to keep val/test evaluation deterministic and comparable across runs
17. **NEW**: paper states "78 subjects, 200 recordings" for Sleep-EDF in Table 1, but the actual Sleep Cassette release is 78 subjects / 153 recordings — this number does not appear to be achievable with any known Sleep-EDF release; treated as a likely error in the manuscript, we proceed with the real 153-recording set

## Immediate next step for you

Run the 2-step quick test above with `--subjects 0-2` and `--limit 6`, and
paste me the console output (epoch counts, any warnings/errors, final saved
array shape). That tells us whether the pipeline actually works against real
EDF files before we scale to all 78 subjects or start writing the model code.
