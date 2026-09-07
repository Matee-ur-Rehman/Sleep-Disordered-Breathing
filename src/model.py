"""
SPRNet (Sleep Pattern Recognition Network), Sec 3.3 of the paper.

Architecture, per-component mapping to the paper:
  1. Modality-specific feature extraction (Eq 6, 9, 10, 11):
     Conv1D(filters=64, kernel_size=3, stride=1) per modality (EEG/EOG/EMG/Airflow),
     followed by global average pooling to a fixed-size feature vector per epoch.
     [ASSUMED] The paper gives conv hyperparameters but not depth (how many conv
     layers) or the pooling/flattening method to go from a variable-length
     conv output to the fixed-size vector F_EEG etc. used downstream. We use
     3 conv layers (with BatchNorm+ReLU) then global average pooling - a
     standard, unremarkable choice, but not literally specified by the paper.

     ECG (Eq 7, 8) is handled differently per the paper: HR/HRV scalar
     features, not a conv branch. We implement this as a small 2-input MLP
     projected to the same feature dimensionality as the conv branches, so
     it can be concatenated alongside them. NOT exercised on Sleep-EDF
     (no ECG channel available); only relevant once SHHS is in hand.

  2. Multimodal fusion: paper's Fig. 1 shows a "Multimodal Feature Fusion"
     box with no equations. [ASSUMED] We use simple concatenation across
     whichever modality branches are present for a given dataset, followed
     by a linear projection to a fixed fusion dimensionality.

  3. BiLSTM temporal modeling (Eq 12): 2 layers, hidden size 128, dropout 0.3,
     bidirectional (paper says "Bi"LSTM). [ASSUMED] paper's H is described as
     R^{T x D_LSTM} - ambiguous whether D_LSTM=128 (one direction) or 256
     (both directions concatenated, the PyTorch default for bidirectional
     LSTMs). We use the standard PyTorch behavior: D_LSTM = 2*hidden_size = 256.

  4. Attention (Eq 13-14 vs. Sec 4.4's "4 heads, embedding dim 128"):
     [AMBIGUITY, RESOLVED] These two parts of the paper describe inconsistent
     mechanisms - Eq 13-14 is a simple single-query additive/dot attention
     producing one scalar weight per time step, with no role for "4 heads"
     or "embedding dim" at all. Sec 4.4 separately states multi-head
     attention with 4 heads and embedding dim 128. We implement the latter
     literally (standard nn.MultiheadAttention, self-attention over the time
     dimension), since it is the more specific/actionable description, and
     project the BiLSTM's 256-dim output down to 128 first to match the
     stated embedding dimension.

  5. Temporal consistency loss (Eq 16): mean squared difference between
     consecutive attended time steps, computed on the post-attention
     representation H'.

  6. Dual output heads (Eq 17, 18): softmax over 5 sleep stages, sigmoid over
     binary respiratory-event probability, both from the same shared H'.
     The respiratory head is defined but not trainable on Sleep-EDF (no
     event labels there); it becomes active once SHHS is available.

INPUT SHAPE CONVENTION:
  The model operates on a SEQUENCE of consecutive 30s epochs (not a single
  epoch in isolation), since sleep-stage context and temporal modeling
  require this. Input: x of shape (batch, T, n_channels, n_samples_per_epoch)
  where T = sequence length (number of consecutive epochs), n_channels =
  total channels across all modalities (e.g. 5 for Sleep-EDF), n_samples =
  EPOCH_SEC * sample_rate (e.g. 3000 for 30s @ 100Hz).
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import sprnet_cfg, SPRNetConfig


class ConvModalityBranch(nn.Module):
    """
    Conv1D feature extractor for one modality (EEG, EOG, EMG, or Airflow).
    Input: (batch*T, n_channels_for_this_modality, n_samples)
    Output: (batch*T, feature_dim)
    """

    def __init__(self, in_channels: int, feature_dim: int, kernel_size: int, stride: int):
        super().__init__()
        # [ASSUMED] 3-layer conv stack with increasing/stable channel width,
        # BatchNorm+ReLU, ending in global average pooling. Paper specifies
        # only filters=64, kernel=3, stride=1 for "the" conv layer (singular),
        # not depth.
        padding = kernel_size // 2  # 'same'-ish padding given odd kernel size
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, feature_dim, kernel_size, stride, padding=padding),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(feature_dim, feature_dim, kernel_size, stride, padding=padding),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(feature_dim, feature_dim, kernel_size, stride, padding=padding),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (batch*T, in_channels, n_samples)
        h = self.net(x)                 # (batch*T, feature_dim, n_samples')
        h = self.pool(h).squeeze(-1)    # (batch*T, feature_dim)
        return h


class ECGBranch(nn.Module):
    """
    ECG branch per Eq 7-8: HR and HRV scalar features (already computed
    upstream in preprocessing, NOT raw ECG waveform), projected to the same
    feature_dim as the conv branches so it can be concatenated alongside
    them. Input here is the 2 precomputed scalars per epoch, not a signal.

    NOT exercised on Sleep-EDF (no ECG channel). Included so the same model
    class is ready for SHHS without modification.
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, x):
        # x: (batch*T, 2)  -- [HR, HRV] per epoch
        return self.net(x)


class SPRNet(nn.Module):
    """
    Full SPRNet: modality branches -> fusion -> BiLSTM -> attention -> dual heads.

    `modality_channel_counts` is a dict like {"eeg": 2, "eog": 1, "emg": 1,
    "airflow": 1} (channel counts per modality actually present in this
    dataset - built at data-loading time via config.build_channel_to_modality_map).
    "ecg" is a special case: if present, expects a separate (batch, T, 2)
    HR/HRV tensor rather than a raw-signal tensor, passed via forward()'s
    `ecg_features` argument.
    """

    def __init__(self, modality_channel_counts: dict, cfg: SPRNetConfig = sprnet_cfg,
                 use_ecg: bool = False):
        super().__init__()
        self.cfg = cfg
        self.modalities = list(modality_channel_counts.keys())
        self.use_ecg = use_ecg

        self.branches = nn.ModuleDict({
            modality: ConvModalityBranch(
                in_channels=n_ch,
                feature_dim=cfg.conv_filters,
                kernel_size=cfg.conv_kernel_size,
                stride=cfg.conv_stride,
            )
            for modality, n_ch in modality_channel_counts.items()
        })

        n_branches = len(self.branches) + (1 if use_ecg else 0)
        if use_ecg:
            self.ecg_branch = ECGBranch(cfg.conv_filters)

        fused_dim_in = n_branches * cfg.conv_filters

        # [ASSUMED] Fusion = concatenation + linear projection down to the
        # BiLSTM's expected input size. Paper's Fig 1 fusion box has no
        # equations; this is the simplest reasonable choice.
        self.fusion = nn.Linear(fused_dim_in, fused_dim_in)

        self.bilstm = nn.LSTM(
            input_size=fused_dim_in,
            hidden_size=cfg.lstm_hidden_size,
            num_layers=cfg.lstm_num_layers,
            dropout=cfg.lstm_dropout if cfg.lstm_num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        lstm_out_dim = cfg.lstm_hidden_size * 2  # bidirectional concat

        # Project down to the attention module's stated embedding dim (128),
        # per Sec 4.4 - see module docstring for why this projection exists.
        self.pre_attn_proj = nn.Linear(lstm_out_dim, cfg.attention_embed_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=cfg.attention_embed_dim,
            num_heads=cfg.attention_heads,
            batch_first=True,
        )

        self.stage_head = nn.Linear(cfg.attention_embed_dim, cfg.num_stages)
        self.disorder_head = nn.Linear(cfg.attention_embed_dim, 1)

    def forward(self, x: dict, ecg_features: torch.Tensor = None):
        """
        x: dict mapping modality name -> tensor of shape
           (batch, T, n_channels_for_modality, n_samples)
        ecg_features: optional (batch, T, 2) tensor of [HR, HRV], required
           iff self.use_ecg is True.

        Returns:
            stage_logits: (batch, T, num_stages)
            disorder_logits: (batch, T, 1)
            h_prime: (batch, T, attention_embed_dim) - post-attention
                     representation, needed externally for the temporal
                     consistency loss (Eq 16).
        """
        any_modality = next(iter(x.values()))
        batch, T = any_modality.shape[0], any_modality.shape[1]

        branch_outputs = []
        for modality, branch in self.branches.items():
            xt = x[modality]  # (batch, T, n_ch, n_samples)
            n_ch, n_samples = xt.shape[2], xt.shape[3]
            xt = xt.reshape(batch * T, n_ch, n_samples)
            feat = branch(xt)                       # (batch*T, feature_dim)
            feat = feat.reshape(batch, T, -1)        # (batch, T, feature_dim)
            branch_outputs.append(feat)

        if self.use_ecg:
            assert ecg_features is not None, "use_ecg=True but no ecg_features passed"
            ecg_flat = ecg_features.reshape(batch * T, 2)
            ecg_feat = self.ecg_branch(ecg_flat).reshape(batch, T, -1)
            branch_outputs.append(ecg_feat)

        fused = torch.cat(branch_outputs, dim=-1)   # (batch, T, n_branches*feature_dim)
        fused = self.fusion(fused)                   # (batch, T, n_branches*feature_dim)

        h, _ = self.bilstm(fused)                     # (batch, T, 2*hidden_size)
        h = self.pre_attn_proj(h)                      # (batch, T, attention_embed_dim)

        h_prime, _ = self.attention(h, h, h)           # self-attention over time; (batch, T, embed_dim)

        stage_logits = self.stage_head(h_prime)         # (batch, T, num_stages)
        disorder_logits = self.disorder_head(h_prime)   # (batch, T, 1)

        return stage_logits, disorder_logits, h_prime


def temporal_consistency_loss(h_prime: torch.Tensor) -> torch.Tensor:
    """
    Eq 16: L_temp = (1/T) * sum_t ||H'_t - H'_{t+1}||^2, averaged over batch.
    h_prime: (batch, T, embed_dim)
    """
    diffs = h_prime[:, 1:, :] - h_prime[:, :-1, :]     # (batch, T-1, embed_dim)
    sq_norms = (diffs ** 2).sum(dim=-1)                # (batch, T-1)
    return sq_norms.mean()