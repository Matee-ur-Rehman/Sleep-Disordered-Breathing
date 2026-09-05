"""
Preprocessing for Sleep-EDF (Sleep Cassette), following Sec 4.2.2 of the paper
as closely as the paper's description allows.

Paper's stated steps (Sec 4.2.2), and how each is implemented here:
  1. Remove duplicate records                -> not applicable to Sleep-EDF (no dup PSGs)
  2. Linear-interpolate missing values        -> implemented (interpolate NaNs per channel)
  3. Remove amplitude outlier segments        -> implemented: any 30s epoch where a channel's
                                                  samples exceed 3 SD from that channel's
                                                  whole-recording mean is flagged/excluded.
  4. Z-score normalize within each channel    -> implemented (per-recording, per-channel)
  5. Time-align multimodal signals            -> implemented via MNE resampling to a common
                                                  rate before windowing (see RESAMPLE_HZ)
  6. Segment into 30s windows, 15s stride     -> see config.USE_OVERLAP; default path uses
                                                  non-overlapping 30s epochs aligned 1:1 with
                                                  hypnogram labels (see config.py note).
  7. Downsample negative class for ECG/resp   -> N/A on Sleep-EDF (no event labels here;
                                                  this only applies to SHHS respiratory task)
  8. Gaussian noise injection on airflow/EMG  -> implemented as an optional augmentation
                                                  function, applied only to TRAINING data,
                                                  not val/test (paper doesn't specify this
                                                  train-only restriction explicitly, but
                                                  injecting noise into val/test would corrupt
                                                  evaluation, so this is our necessary
                                                  interpretation, flagged here).

IMPORTANT: This script assumes Sleep-EDF SC files are on disk under
`--data_dir`, in MNE's fetch_data output layout (each subject/night has a
*-PSG.edf and matching *-Hypnogram.edf). Run scripts/download_sleep_edf.py
first.
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    SLEEP_EDF_CHANNELS, STAGE_TO_IDX, EPOCH_SEC, AMPLITUDE_OUTLIER_SD,
    NOISE_STD_AIRFLOW_EMG, NOISE_MEAN, WAKE_TRIM_EPOCHS,
)

RESAMPLE_HZ = 100  # [ASSUMED] common target rate; EEG/EOG are natively 100Hz in
                    # Sleep-EDF, EMG/airflow are natively 1Hz, so this upsamples
                    # the slow channels to align all modalities in time. The
                    # paper does not specify a target rate for cross-modality
                    # alignment; 100Hz (matching the fastest native channel) is
                    # the least lossy choice available.

# Sleep-EDF hypnogram annotation strings -> our 5-class scheme.
# Sleep-EDF uses R&K staging (W, 1, 2, 3, 4, R, M, ?); we merge 3+4 -> N3 and
# drop Movement/Unknown epochs, per standard practice (paper doesn't specify
# for Sleep-EDF; AASM-vs-R&K merging of N3/N4 is the universal convention used
# in every reproduction we found in the literature search, e.g. DeepSleepNet-Lite).
ANNOTATION_TO_STAGE = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
    # "Sleep stage ?" and "Movement time" are intentionally NOT mapped;
    # such epochs are dropped.
}


def load_recording(psg_path: str, hyp_path: str):
    """Load one PSG + hypnogram pair via MNE, return raw signals + per-epoch labels."""
    import mne

    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    annot = mne.read_annotations(hyp_path)
    raw.set_annotations(annot, emit_warning=False)

    return raw


def extract_epoch_labels(raw, epoch_sec: int = EPOCH_SEC):
    """
    Walk the raw's annotations and produce a list of (onset_sec, duration_sec, stage_str)
    for every 30s sub-block covered by a scored annotation. Un-mapped stages (Movement,
    Unknown) are skipped.
    """
    epochs = []
    for annot in raw.annotations:
        stage = ANNOTATION_TO_STAGE.get(annot["description"])
        if stage is None:
            continue
        onset = annot["onset"]
        duration = annot["duration"]
        n_sub_epochs = int(round(duration / epoch_sec))
        for k in range(n_sub_epochs):
            epochs.append((onset + k * epoch_sec, epoch_sec, stage))
    return epochs


def pick_available_channels(raw, channel_candidates):
    """Return the subset of requested channel names actually present in this recording."""
    available = set(raw.ch_names)
    return [ch for ch in channel_candidates if ch in available]


def clean_and_normalize(raw, channels):
    """
    Apply Sec 4.2.2 cleaning: linear interpolation of NaNs, amplitude-outlier
    exclusion (flagged, not removed from raw here - handled at epoch level in
    build_epoch_array), and per-channel z-score normalization.
    """
    data = raw.get_data(picks=channels)  # shape (n_channels, n_samples)

    # 1. Interpolate NaNs (linear) per channel.
    for i in range(data.shape[0]):
        row = data[i]
        nan_mask = np.isnan(row)
        if nan_mask.any():
            valid_idx = np.flatnonzero(~nan_mask)
            if len(valid_idx) >= 2:
                row[nan_mask] = np.interp(np.flatnonzero(nan_mask), valid_idx, row[valid_idx])
            else:
                row[nan_mask] = 0.0  # degenerate case: whole channel missing
            data[i] = row

    # 2. Per-channel z-score normalization (paper Sec 4.2.2).
    means = data.mean(axis=1, keepdims=True)
    stds = data.std(axis=1, keepdims=True)
    stds[stds == 0] = 1.0  # avoid div-by-zero on flat channels
    data_norm = (data - means) / stds

    return data_norm, means.squeeze(-1), stds.squeeze(-1)


def build_epoch_array(data_norm, sfreq, epoch_labels, channels_order):
    """
    Slice the continuous, normalized signal into non-overlapping 30s epochs
    matching the hypnogram labels, then apply Sec 4.2.2's amplitude-outlier
    exclusion.

    [AMBIGUITY / DEVIATION FROM LITERAL PAPER TEXT, FLAGGED]
    The paper says outliers are "identified by amplitude thresholds exceeding
    three standard deviations from the channel mean." Taken literally at the
    per-sample level, this would flag almost every 30s epoch: with ~3000
    samples/channel/epoch, pure Gaussian noise alone has a high chance of at
    least one sample beyond 3 SD just by chance, and real EEG transients
    (K-complexes, blinks) trigger it even more often. A per-sample criterion
    would discard nearly the entire dataset, which cannot be what the authors
    intended (their Table 1 stage-distribution stats are computed on "most"
    of the data, implying only a small fraction is excluded).
    Our resolution: treat "abnormal segment" at the EPOCH level, using each
    epoch's RMS amplitude per channel, and flag an epoch as an outlier if its
    RMS is more than AMPLITUDE_OUTLIER_SD standard deviations from the mean
    RMS across all epochs of that channel in the SAME recording. This targets
    genuinely anomalous segments (sensor artifacts/disconnection) rather than
    ordinary physiological transients, matching the spirit of the paper's
    described step, but is our own choice of aggregation and should be
    revisited/reported as an assumption in any write-up.

    Returns:
        X: np.ndarray, shape (n_valid_epochs, n_channels, n_samples_per_epoch)
        y: np.ndarray, shape (n_valid_epochs,) of integer stage labels
        excluded_count: int
    """
    n_samples_per_epoch = int(round(EPOCH_SEC * sfreq))
    total_samples = data_norm.shape[1]

    # First pass: slice all candidate epochs (before outlier filtering) so we
    # can compute per-channel RMS statistics across the whole recording.
    candidates = []  # list of (seg, stage)
    for onset_sec, duration_sec, stage in epoch_labels:
        start = int(round(onset_sec * sfreq))
        end = start + n_samples_per_epoch
        if end > total_samples:
            continue
        candidates.append((data_norm[:, start:end], stage))

    if not candidates:
        return np.empty((0, data_norm.shape[0], n_samples_per_epoch)), np.empty((0,), dtype=int), 0

    n_channels = data_norm.shape[0]
    rms_per_epoch = np.array([
        np.sqrt(np.mean(seg ** 2, axis=1)) for seg, _ in candidates
    ])  # shape (n_epochs, n_channels)

    rms_mean = rms_per_epoch.mean(axis=0)
    rms_std = rms_per_epoch.std(axis=0)
    rms_std[rms_std == 0] = 1.0

    z_rms = np.abs(rms_per_epoch - rms_mean) / rms_std  # (n_epochs, n_channels)
    is_outlier = np.any(z_rms > AMPLITUDE_OUTLIER_SD, axis=1)  # (n_epochs,)

    X_list, y_list = [], []
    excluded = 0
    for (seg, stage), outlier in zip(candidates, is_outlier):
        if outlier:
            excluded += 1
            continue
        X_list.append(seg)
        y_list.append(STAGE_TO_IDX[stage])

    if not X_list:
        return np.empty((0, n_channels, n_samples_per_epoch)), np.empty((0,), dtype=int), excluded

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=int)
    return X, y, excluded


def trim_wake_padding(X, y, buffer_epochs=WAKE_TRIM_EPOCHS):
    """
    Trim long wake ('W') padding at the start and end of a whole-night
    recording, keeping only `buffer_epochs` of W immediately before the first
    non-W epoch and after the last non-W epoch. See config.py WAKE_TRIM_EPOCHS
    for why this exists - the paper does not describe this step, but without
    it Sleep-EDF's continuous ~20-24h recordings are dominated by W (~69% in
    our own empirical test), far above the paper's reported 30.1% (Table 1).

    If a recording is all-W (e.g. failed scoring) or has no non-W epochs,
    it is returned unchanged (nothing sensible to trim around).
    """
    wake_idx = STAGE_TO_IDX["W"]
    non_wake_positions = np.flatnonzero(y != wake_idx)
    if len(non_wake_positions) == 0:
        return X, y

    first_sleep = non_wake_positions[0]
    last_sleep = non_wake_positions[-1]

    start = max(0, first_sleep - buffer_epochs)
    end = min(len(y), last_sleep + buffer_epochs + 1)

    return X[start:end], y[start:end]


def inject_gaussian_noise(X, channel_indices, mean=NOISE_MEAN, std=NOISE_STD_AIRFLOW_EMG, rng=None):
    """
    Add Gaussian noise (mu=0, sigma=0.01 per paper) to specified channel
    indices (airflow, EMG) - intended for TRAINING data only. Paper doesn't
    explicitly restrict this to train-only, but doing it on val/test would
    make evaluation non-deterministic and non-comparable across runs, so we
    apply it at train-time only. Flagged as our interpretation.
    """
    rng = rng or np.random.default_rng()
    X_noisy = X.copy()
    for ci in channel_indices:
        X_noisy[:, ci, :] += rng.normal(mean, std, size=X_noisy[:, ci, :].shape)
    return X_noisy


def process_one_recording(psg_path: str, hyp_path: str, verbose: bool = True):
    """Full pipeline for a single subject-night: load -> pick channels -> clean ->
    normalize -> epoch -> return arrays + channel order used."""
    import mne
    mne.set_log_level("ERROR")

    raw = load_recording(psg_path, hyp_path)

    channel_order = []
    for modality in ("eeg", "eog", "emg", "airflow"):
        found = pick_available_channels(raw, SLEEP_EDF_CHANNELS[modality])
        channel_order.extend(found)

    if not channel_order:
        raise RuntimeError(f"No expected channels found in {psg_path}. "
                            f"Available: {raw.ch_names}")

    # Resample all channels to a common rate for alignment (paper Sec 4.2.2:
    # "aligned in time to synchronize multimodal sequences").
    if abs(raw.info["sfreq"] - RESAMPLE_HZ) > 1e-6:
        raw.resample(RESAMPLE_HZ, npad="auto", verbose=False)

    epoch_labels = extract_epoch_labels(raw)
    if not epoch_labels:
        warnings.warn(f"No usable annotated epochs in {hyp_path}")

    data_norm, means, stds = clean_and_normalize(raw, channel_order)
    X, y, excluded = build_epoch_array(data_norm, RESAMPLE_HZ, epoch_labels, channel_order)

    n_before_trim = X.shape[0]
    X, y = trim_wake_padding(X, y)
    n_trimmed = n_before_trim - X.shape[0]

    if verbose:
        print(f"  {Path(psg_path).name}: {n_before_trim} epochs after outlier removal "
              f"({excluded} excluded), {n_trimmed} trimmed as excess wake padding, "
              f"{X.shape[0]} final epochs kept. channels={channel_order}")

    return X, y, channel_order


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Directory containing downloaded Sleep-EDF SC files.")
    parser.add_argument("--out", type=str, default="./data/processed/sleep_edf_sc",
                         help="Output DIRECTORY (not a single file) for sharded .npz output. "
                              "One shard per successfully processed recording is written here, "
                              "plus a manifest.npz listing all shards.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Optional: only process the first N PSG/hyp pairs (for a quick test run).")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    psg_files = sorted(data_dir.rglob("*PSG.edf"))
    if args.limit:
        psg_files = psg_files[: args.limit]

    if not psg_files:
        print(f"No PSG files found under {data_dir}. Did you run download_sleep_edf.py first?",
              file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # [FIX, replaces earlier in-memory-concatenate design]
    # The original version accumulated every recording's array in a Python
    # list and concatenated them all into ONE array at the end. For the full
    # 153-recording Sleep-EDF SC set this required ~5.7 GiB in a single
    # contiguous allocation (plus a temporary ~2x spike during concatenate),
    # which crashed on a normal desktop/laptop RAM budget - confirmed
    # empirically (numpy._core._exceptions._ArrayMemoryError at the final
    # concatenate step, plus a cascade of per-file allocation failures
    # leading up to it from accumulated memory pressure).
    # Fix: write each recording's array to its own small shard file on disk
    # IMMEDIATELY after processing it, then let Python garbage-collect it
    # before moving to the next recording. A lightweight manifest.npz records
    # which shards exist and in what order, without ever holding more than
    # one recording's data in memory at a time.
    shard_paths = []
    total_epochs = 0
    channel_order_ref = None
    n_success, n_failed = 0, 0

    for psg_path in psg_files:
        stem = psg_path.name.replace("-PSG.edf", "")
        subject_night_prefix = stem[:6]
        candidates = list(psg_path.parent.glob(f"{subject_night_prefix}*Hypnogram.edf"))
        if not candidates:
            print(f"  WARNING: no matching hypnogram for {psg_path.name}, skipping.")
            n_failed += 1
            continue
        hyp_path = candidates[0]

        try:
            X, y, channel_order = process_one_recording(str(psg_path), str(hyp_path))
        except Exception as e:
            print(f"  ERROR processing {psg_path.name}: {e}")
            n_failed += 1
            continue

        if X.shape[0] == 0:
            n_failed += 1
            continue

        if channel_order_ref is None:
            channel_order_ref = channel_order
        elif channel_order != channel_order_ref:
            print(f"  WARNING: channel set mismatch for {psg_path.name} "
                  f"({channel_order} vs {channel_order_ref}); skipping to keep shards consistent.")
            n_failed += 1
            continue

        # [FIX] float32 instead of float64: halves memory footprint. Z-scored
        # data (mean 0, std 1 by construction) does not need float64
        # precision; float32's ~7 significant digits is far more than enough.
        X = X.astype(np.float32)

        subject_id = subject_night_prefix[3:5]
        shard_name = f"{subject_night_prefix}.npz"
        shard_path = out_dir / shard_name
        np.savez_compressed(shard_path, X=X, y=y,
                             subject=np.array([subject_id] * X.shape[0]))

        shard_paths.append(shard_name)
        total_epochs += X.shape[0]
        n_success += 1
        print(f"  -> wrote shard {shard_name} ({X.shape[0]} epochs)")

        # Explicitly drop references so this recording's memory is freed
        # before the next iteration allocates a new one.
        del X, y

    if n_success == 0:
        print("No recordings successfully processed.", file=sys.stderr)
        sys.exit(1)

    manifest_path = out_dir / "manifest.npz"
    np.savez(manifest_path,
             shard_files=np.array(shard_paths),
             channels=np.array(channel_order_ref),
             total_epochs=total_epochs)

    print(f"\nDone. {n_success} recordings processed successfully, {n_failed} failed/skipped.")
    print(f"Total epochs across all shards: {total_epochs}")
    print(f"Channels used: {channel_order_ref}")
    print(f"Shards + manifest written to: {out_dir}")
    print("\nNOTE: output is now a DIRECTORY of per-recording shard files, not one "
          "single .npz. Use scripts/check_label_distribution.py (updated) or the "
          "future dataset loader to read across all shards without loading "
          "everything into RAM at once.")


if __name__ == "__main__":
    main()