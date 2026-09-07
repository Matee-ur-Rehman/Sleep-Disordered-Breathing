"""
Central configuration for the SDB / SPRNet + IDSSA reproduction.

Every value below is tagged as either:
  [PAPER]      - explicitly stated in the manuscript (Zhang et al., Sci Rep, in press,
                 s41598-026-56111-6), with the source subsection/equation noted.
  [ASSUMED]    - NOT specified in the paper. We picked a reasonable value and flagged
                 it here so it is easy to find and revisit. Do not treat these as
                 "the paper's setting" when writing up results.
  [DERIVED]    - computed from the dataset itself (e.g. transition matrix), as the
                 paper describes but without giving the actual numbers.
"""

from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Sleep stages (Sec 3.2)  [PAPER]
# ---------------------------------------------------------------------------
SLEEP_STAGES = ["W", "N1", "N2", "N3", "REM"]
STAGE_TO_IDX = {s: i for i, s in enumerate(SLEEP_STAGES)}

# ---------------------------------------------------------------------------
# EEG frequency bands used for PSD features (Sec 3.2, Eq. 2)  [PAPER]
# ---------------------------------------------------------------------------
EEG_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}
# Gamma band explicitly excluded per Sec 3.2 (noise/muscle artifact susceptibility). [PAPER]


# ---------------------------------------------------------------------------
# Windowing (Sec 4.2.2)
# ---------------------------------------------------------------------------
EPOCH_SEC = 30          # [PAPER] standard AASM epoch length, used for labels
WINDOW_SEC = 30         # [PAPER] "fixed length windows of 30 seconds"
STRIDE_SEC = 15         # [PAPER] "with a stride of 15 seconds"
# NOTE [ASSUMED / RESOLVED AMBIGUITY]:
# The paper uses overlapping 30s windows at 15s stride for the *sequential feature
# extraction* going into the BiLSTM, but AASM sleep-stage labels are defined on
# non-overlapping 30s epochs. The paper does not explain how overlapping windows are
# reconciled with one-label-per-epoch evaluation. Our resolution: we build the model
# and dataset for the NON-OVERLAPPING 30s epoch case first (stride = epoch length),
# because that is what is directly comparable to standard sleep-staging evaluation
# and is unambiguous. We keep STRIDE_SEC configurable so we can later add the
# overlapping-window variant if we want to probe whether it changes results, but the
# main reproduction path defaults to NON-overlapping epochs (see USE_OVERLAP below).
USE_OVERLAP = False     # [ASSUMED] see note above; True would follow the literal
                         # 15s-stride text but breaks the 1-label-per-epoch mapping.

NOISE_STD_AIRFLOW_EMG = 0.01   # [PAPER] Gaussian noise sigma injected into airflow/EMG
NOISE_MEAN = 0.0                # [PAPER]

AMPLITUDE_OUTLIER_SD = 3.0      # [PAPER] segments beyond 3 SD from channel mean excluded

# [NEW ASSUMPTION #18] Wake-period trimming.
# Sleep-EDF Sleep Cassette recordings are continuous ~20-24 hour captures that
# include long wake periods before/after the actual sleep episode. Verified
# empirically on our own test batch: without trimming, W epochs make up 68.7%
# of the data, vs. the paper's reported 30.1% (Table 1) - a ~2.3x mismatch.
# The paper's Sec 4.2.2 does NOT mention any wake-trimming step at all, but
# achieving anything close to their reported stage distribution requires one.
# This is standard practice across the Sleep-EDF literature (e.g. Supratak et
# al. 2017, DeepSleepNet, trims to 30 minutes of wake before/after the sleep
# period) even though this specific paper is silent on it.
# [ASSUMED] We adopt the same 30-minute buffer as the most common convention,
# while flagging that the paper gives us no basis to confirm this exact value
# reproduces their 30.1% figure - we will check empirically once applied.
WAKE_TRIM_MINUTES = 30
WAKE_TRIM_EPOCHS = int(WAKE_TRIM_MINUTES * 60 / EPOCH_SEC)  # in units of 30s epochs


# ---------------------------------------------------------------------------
# Dataset-specific channel maps.
# Sleep-EDF Sleep Cassette (SC) does NOT have ECG. It DOES have an oro-nasal
# respiration channel in most SC recordings (verified against PhysioNet's own
# dataset description), so we can exercise EEG/EOG/EMG/Airflow branches on
# Sleep-EDF, but NOT the ECG branch (HR/HRV), which is SHHS-only. [PAPER + verified]
# ---------------------------------------------------------------------------
SLEEP_EDF_CHANNELS = {
    "eeg": ["EEG Fpz-Cz", "EEG Pz-Oz"],
    "eog": ["EOG horizontal"],
    "emg": ["EMG submental"],
    "airflow": ["Resp oro-nasal"],   # not present in 100% of SC recordings; handled
                                       # as optional in the loader.
}


def build_channel_to_modality_map(channel_order, channel_groups):
    """
    Given the actual list of channel names present in a recording (as saved
    in manifest.npz's 'channels' field) and a modality->channel-name-list
    mapping (e.g. SLEEP_EDF_CHANNELS), return a dict {modality: [indices]}
    giving the row-indices into the data array that belong to each modality.

    This lets the model dynamically build one branch per modality actually
    present, rather than hardcoding channel counts/positions - important
    since Sleep-EDF and SHHS will have different channel sets (SHHS adds
    ECG, Sleep-EDF does not), and even within Sleep-EDF SC not every
    recording necessarily has every channel (e.g. airflow can be missing).
    """
    modality_indices = {}
    for modality, names in channel_groups.items():
        idx = [i for i, ch in enumerate(channel_order) if ch in names]
        if idx:
            modality_indices[modality] = idx
    return modality_indices

# SHHS channel names differ by dataset version (shhs1/shhs2 EDFs use varying header
# labels across the cohort). [ASSUMED - to be confirmed once SHHS access is granted
# and we can inspect actual EDF headers] Placeholder names below, DO NOT trust yet.
SHHS_CHANNELS_PLACEHOLDER = {
    "eeg": ["EEG", "EEG2"],
    "eog": ["EOG(L)", "EOG(R)"],
    "emg": ["EMG"],
    "ecg": ["ECG"],
    "airflow": ["AIRFLOW"],
}


# ---------------------------------------------------------------------------
# SPRNet architecture (Sec 3.3 / Sec 4.4)
# ---------------------------------------------------------------------------
@dataclass
class SPRNetConfig:
    conv_filters: int = 64        # [PAPER] Sec 4.4
    conv_kernel_size: int = 3     # [PAPER]
    conv_stride: int = 1          # [PAPER]

    lstm_hidden_size: int = 128   # [PAPER] "hidden size of 128"
    lstm_num_layers: int = 2      # [PAPER] "two layers"
    lstm_dropout: float = 0.3     # [PAPER]

    attention_heads: int = 4      # [PAPER]
    attention_embed_dim: int = 128  # [PAPER]

    attention_context_window: int = 3   # [ASSUMED] tau in Eq. 15, not given numerically.
                                          # Picking +-3 time steps (i.e. 7-step local
                                          # context) as a modest default; revisit.

    task_loss_lambda: float = 1.0        # [ASSUMED] lambda in Eq. 21 (L_stage + lambda*L_disorder)
                                          # not specified. Starting at 1.0 (equal weighting).

    num_stages: int = len(SLEEP_STAGES)


# ---------------------------------------------------------------------------
# IDSSA (Sec 3.4)
# ---------------------------------------------------------------------------
@dataclass
class IDSSAConfig:
    # Markov transition matrix adaptation rate, Eq. 23: T_new = (1-eta)*T + eta*T_hat
    # Sec 4.4 mentions "a transition probability matrix ... adjusted with a smoothing
    # factor of 0.1" - we treat this as eta, though the paper never explicitly
    # names it as such. [ASSUMED, weakly supported by Sec 4.4 text]
    eta_adaptation_rate: float = 0.1

    # Regularization strength for Eq. 24 (L_reg on transition matrix). Paper reuses
    # the symbol "lambda" here, DISTINCT from task_loss_lambda above and from the
    # lambda in Eq. 26. Not given a value anywhere. [ASSUMED]
    transition_reg_lambda: float = 0.1

    # Per-stage persistence timescale tau_st in Eq. 25. Not specified per stage.
    # [ASSUMED] Using a single shared timescale (seconds) as a starting point;
    # a more faithful version would fit one tau per stage from empirical stage
    # durations, which we can do once we have real hypnogram statistics.
    persistence_tau_sec: float = 120.0

    # Multi-objective loss weights, Eq. 26: L = L_staging + lambda*L_disorder + beta*L_uncertainty
    # lambda here is presumably meant to equal task_loss_lambda in Eq 21, but the
    # paper does not say so explicitly - another instance of overloaded notation.
    # [ASSUMED] We reuse SPRNetConfig.task_loss_lambda for this lambda to at least
    # be internally consistent, and add a separate beta:
    beta_uncertainty: float = 0.1   # [ASSUMED] not specified anywhere.


# ---------------------------------------------------------------------------
# Training / evaluation protocol (Sec 4.3, Sec 4.4) - these ARE fully specified.
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    epochs: int = 100                 # [PAPER]
    batch_size: int = 64              # [PAPER]
    lr: float = 1e-3                  # [PAPER]
    weight_decay: float = 1e-4        # [PAPER]
    optimizer: str = "adam"           # [PAPER]
    lr_scheduler: str = "cosine_annealing"  # [PAPER]
    early_stopping_patience: int = 10       # [PAPER], monitored on val loss
    seed: int = 42                    # [PAPER]
    n_repeats: int = 5                # [PAPER] "repeated five times using different
                                        # random seeds" - NOTE: this directly conflicts
                                        # with "seed fixed at 42 for reproducibility
                                        # across all runs" elsewhere in the paper.
                                        # [ASSUMED RESOLUTION] we treat n_repeats=5 as
                                        # authoritative for the statistical protocol,
                                        # and use seeds {42, 43, 44, 45, 46} for the 5
                                        # runs, since the paper doesn't give the actual
                                        # 5 seed values.
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44, 45, 46])  # [ASSUMED]

    train_frac: float = 0.8           # [PAPER] 8:1:1 subject-wise split
    val_frac: float = 0.1
    test_frac: float = 0.1
    split_seed: int = 42              # [ASSUMED] paper doesn't give a specific seed
                                        # for the train/val/test SPLIT ASSIGNMENT
                                        # itself (as opposed to model init/training
                                        # seeds, which are separately covered by
                                        # `seeds` above). We fix one so the split is
                                        # at least reproducible across our own runs.


# ---------------------------------------------------------------------------
# Sequence construction for BiLSTM input (not specified by the paper at all)
# ---------------------------------------------------------------------------
@dataclass
class SequenceConfig:
    # [ASSUMED, NEW AMBIGUITY] The paper never states how many consecutive
    # 30s epochs form one training sequence fed to the BiLSTM. We pick 20
    # epochs = 10 minutes of context per sequence as a reasonable middle
    # ground (long enough to give the BiLSTM/attention real temporal context,
    # short enough to keep batches small on CPU) - not derived from the paper.
    sequence_length: int = 20
    # Non-overlapping sequences by default (stride == length), i.e. every
    # epoch appears in exactly one training sequence. [ASSUMED] An
    # overlapping-sequence variant (stride < length) is arguably more
    # paper-literal given their 15s-stride windowing language elsewhere, but
    # we already resolved that ambiguity in favor of non-overlapping epochs
    # for consistency (see config.USE_OVERLAP note above) - keeping
    # non-overlapping sequences here matches that same resolution.
    sequence_stride: int = 20


seq_cfg = SequenceConfig()


sprnet_cfg = SPRNetConfig()
idssa_cfg = IDSSAConfig()
train_cfg = TrainConfig()