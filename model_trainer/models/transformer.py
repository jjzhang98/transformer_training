"""Set Transformer model for Learning-to-Rank on grouped financial data.

Architecture:
    Input (B, N, F) → Linear(F, d_model)
               → N × TransformerEncoderLayer (no positional encoding)
               → LayerNorm
               → Linear(d_model, 1) per position
               → Scores (B, N)

Key properties:
    - Permutation equivariant: reordering candidates within a group
      produces correspondingly reordered scores with identical values.
      This is guaranteed by removing positional encoding — the
      self-attention mechanism is inherently permutation equivariant.
    - Group-aware: each candidate attends to all others in the same group
      via self-attention, enabling relative ranking rather than isolated scoring.
    - Compatible with Pearson / ListNet / ListMLE losses via GroupedDataset.

Training:
    - Data grouped by (date, time_sec, code).
    - Groups padded to the same length within each batch; attention masking
      prevents cross-group information leakage and padding contamination.
    - Early stopping on validation top1_above_mean_rate.
"""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base import BaseModel
from ..dataset import GroupedDataset
from ..data import resolve_device
from ..losses import get_loss_function, is_loss_compilable
from ..metrics import TOPK_VALUES, build_prior_topk_abtypes


def _pad_collate_fn(batch):
    """Pad variable-length groups into a batched tensor for Transformer processing.

    Args:
        batch: List of (features, labels, labels_original, abtype) tuples from GroupedDataset.

    Returns:
        features_padded:    (batch_size, max_group_len, num_features) float32
        labels_padded:      (batch_size, max_group_len) float32
        labels_original_pad: (batch_size, max_group_len) float32
        abtype_padded:      (batch_size, max_group_len) int64
        valid_mask:         (batch_size, max_group_len) bool — True for valid positions
        group_sizes:        list[int]
    """
    max_group_len = max(features.shape[0] for features, _, _, _ in batch)
    num_features = batch[0][0].shape[1]
    batch_size = len(batch)

    features_padded = torch.zeros(batch_size, max_group_len, num_features, dtype=torch.float32)
    labels_padded = torch.zeros(batch_size, max_group_len, dtype=torch.float32)
    labels_original_pad = torch.zeros(batch_size, max_group_len, dtype=torch.float32)
    abtype_padded = torch.full((batch_size, max_group_len), -1, dtype=torch.int64)
    valid_mask = torch.zeros(batch_size, max_group_len, dtype=torch.bool)
    group_sizes: List[int] = []

    for batch_idx, (features, labels, labels_original, abtype) in enumerate(batch):
        group_len = features.shape[0]
        features_padded[batch_idx, :group_len] = features
        labels_padded[batch_idx, :group_len] = labels
        labels_original_pad[batch_idx, :group_len] = labels_original
        abtype_padded[batch_idx, :group_len] = abtype
        valid_mask[batch_idx, :group_len] = True
        group_sizes.append(group_len)

    return features_padded, labels_padded, labels_original_pad, abtype_padded, valid_mask, group_sizes


class _SetTransformerEncoder(nn.Module):
    """Permutation-equivariant Transformer encoder for group-wise scoring.

    No positional encoding → permutation equivariant.
    Self-attention within each group → each candidate sees all others.
    Final LayerNorm for training stability (Pre-LN architecture).

    Args:
        input_dim: Number of input features per candidate.
        d_model: Transformer hidden dimension.
        nhead: Number of attention heads (auto-adjusted if not divisible by d_model).
        num_layers: Number of TransformerEncoder layers.
        dim_ff: Feed-forward dimension; defaults to d_model * 4 if 0.
        dropout: Dropout probability.
        apply_batch_softmax: Whether to apply softmax to output scores within each group.
    """

    def __init__(
            self,
            input_dim: int,
            d_model: int = 256,
            nhead: int = 8,
            num_layers: int = 2,
            dim_ff: int = 0,
            dropout: float = 0.1,
            apply_batch_softmax: bool = False,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            for candidate_nhead in range(nhead, 0, -1):
                if d_model % candidate_nhead == 0:
                    nhead = candidate_nhead
                    break
        dim_ff = dim_ff or d_model * 4

        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.head = nn.Linear(d_model, 1)
        self.apply_batch_softmax = apply_batch_softmax

    def forward(
            self,
            features: torch.Tensor,
            valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the Set Transformer.

        Args:
            features: Input features, shape (batch_size, max_group_len, num_features).
            valid_mask: Boolean mask for valid positions, shape (batch_size, max_group_len);
                        True for valid, False for padding.

        Returns:
            Output scores, shape (batch_size, max_group_len). Padding positions
            are zeroed out (or softmax-normalized if apply_batch_softmax is True).
        """
        hidden = self.input_proj(features)
        key_padding_mask = ~valid_mask if valid_mask is not None else None
        hidden = self.encoder(hidden, src_key_padding_mask=key_padding_mask)
        raw_scores = self.head(hidden).squeeze(-1)

        if self.apply_batch_softmax:
            raw_scores = raw_scores.masked_fill(~valid_mask, float("-inf"))
            raw_scores = torch.softmax(raw_scores, dim=-1)
            raw_scores = raw_scores.masked_fill(~valid_mask, 0.0)
        else:
            raw_scores = raw_scores.masked_fill(~valid_mask, 0.0)

        return raw_scores


class TransformerModel(BaseModel):
    """Set Transformer for Learning-to-Rank on grouped financial data.

    Unlike MLP which scores each sample independently, this model uses
    self-attention so each candidate can see all others in its group,
    producing better relative rankings.

    Permutation equivariant: reordering candidates within a group
    produces correspondingly reordered scores with identical values.

    Args:
        name: Display name for this model instance.
        d_model: Transformer hidden dimension.
        nhead: Number of attention heads.
        num_layers: Number of TransformerEncoder layers.
        dim_ff: Feed-forward dimension; 0 means d_model * 4.
        dropout: Dropout probability.
    """

    def __init__(
            self,
            name: str = "transformer",
            d_model: int = 256,
            nhead: int = 8,
            num_layers: int = 2,
            dim_ff: int = 0,
            dropout: float = 0.1,
    ) -> None:
        super().__init__(name)
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_ff = dim_ff
        self.dropout = dropout
        self._net: Optional[_SetTransformerEncoder] = None
        self._device: Optional[torch.device] = None

    def fit(
            self,
            df_train: pd.DataFrame,
            factor_list: List[str],
            y_col: str,
            df_val: Optional[pd.DataFrame] = None,
            training_cfg=None,
    ) -> dict:
        """Train the Set Transformer model on grouped data.

        Args:
            df_train: Training DataFrame with feature, label, and grouping columns.
            factor_list: List of feature column names.
            y_col: Label column name.
            df_val: Optional validation DataFrame for early stopping.
            training_cfg: TrainingConfig dataclass with hyperparameters.

        Returns:
            History dict with per-epoch metrics for train and validation.
        """
        cfg = training_cfg
        num_epochs = cfg.epochs if cfg else 200
        batch_size = cfg.batch_size if cfg else 512
        learning_rate = cfg.learning_rate if cfg else 3e-4
        early_stopping_patience = cfg.early_stopping_patience if cfg else 30
        min_epochs = cfg.min_epochs if cfg else 30
        min_delta = cfg.min_delta if cfg else 1e-4
        device_str = cfg.device if cfg else "auto"
        loss_type = cfg.loss_type if cfg else "pearson"
        topk = cfg.topk if cfg else 1
        rank_before_softmax = cfg.rank_before_softmax if cfg else True
        use_hard_label = cfg.use_hard_label if cfg else False
        label_smoothing = cfg.label_smoothing if cfg else 0.0

        use_listnet = loss_type == "listnet"
        use_listmle = loss_type == "listmle"
        use_torch_compile = cfg.torch_compile if cfg else False

        self._device = resolve_device(device_str)
        use_pin_memory = self._device.type == "cuda"
        num_features = len(factor_list)

        self._net = _SetTransformerEncoder(
            input_dim=num_features,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_ff=self.dim_ff,
            dropout=self.dropout,
            apply_batch_softmax=use_listnet,
        ).to(self._device)

        if use_torch_compile and self._device.type != "cpu":
            self._net = torch.compile(self._net)

        total_params = sum(p.numel() for p in self._net.parameters())
        trainable_params = sum(
            p.numel() for p in self._net.parameters() if p.requires_grad
        )
        print(f"\n{'=' * 60}")
        print(f"All Hyperparameters: {self.name} (transformer)")
        print(f"{'=' * 60}")
        print(f"  [Training]")
        print(f"    epochs: {num_epochs}")
        print(f"    batch_size: {batch_size}")
        print(f"    learning_rate: {learning_rate}")
        print(f"    loss_type: {loss_type}")
        print(f"    early_stopping_patience: {early_stopping_patience}")
        print(f"    rank_before_softmax: {rank_before_softmax}")
        print(f"    use_hard_label: {use_hard_label}")
        print(f"    label_smoothing: {label_smoothing}")
        print(f"    torch_compile: {use_torch_compile}")
        print(f"  [Model]")
        print(f"    d_model: {self.d_model}")
        print(f"    nhead: {self.nhead}")
        print(f"    num_layers: {self.num_layers}")
        print(f"    dim_ff: {self.dim_ff or self.d_model * 4}")
        print(f"    dropout: {self.dropout}")
        print(f"    apply_batch_softmax: {use_listnet}")
        print(f"  [Network]")
        print(f"    Input features: {num_features}")
        print(f"    Total parameters: {total_params:,}")
        print(f"    Trainable parameters: {trainable_params:,}")
        print(f"    Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
        print(f"    Device: {self._device}")
        print(f"{'=' * 60}\n")

        prior_topk_abtypes: Optional[Dict[int, np.ndarray]] = None
        has_abtype = "abtype" in df_train.columns
        if has_abtype:
            df_union = (
                pd.concat([df_train, df_val], ignore_index=True)
                if df_val is not None and not df_val.empty
                else df_train
            )
            prior_topk_abtypes = build_prior_topk_abtypes(df=df_union, y_col=y_col)

        train_dataset = GroupedDataset(
            df=df_train,
            factor_list=factor_list,
            y_col=y_col,
            apply_softmax_to_y=use_listnet,
            rank_before_softmax=rank_before_softmax,
            topk=topk,
            use_hard_label=use_hard_label,
            label_smoothing=label_smoothing,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=use_pin_memory,
            collate_fn=_pad_collate_fn,
        )

        val_loader = None
        if df_val is not None and not df_val.empty:
            val_dataset = GroupedDataset(
                df=df_val,
                factor_list=factor_list,
                y_col=y_col,
                apply_softmax_to_y=use_listnet,
                rank_before_softmax=rank_before_softmax,
                topk=topk,
                use_hard_label=use_hard_label,
                label_smoothing=label_smoothing,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size * 4,
                shuffle=False,
                num_workers=4,
                pin_memory=use_pin_memory,
                collate_fn=_pad_collate_fn,
            )

        criterion = get_loss_function(loss_type=loss_type)
        if use_torch_compile and self._device.type != "cpu" and is_loss_compilable(loss_type=loss_type):
            criterion = torch.compile(criterion)
        optimizer = torch.optim.Adam(
            self._net.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=num_epochs, eta_min=learning_rate * 0.01
        )

        history = {
            "train_loss": [], "val_loss": [],
            "train_topk_hit_rate": [], "train_excess_hit_rate": [], "train_avg_pred_rank": [],
            "val_topk_hit_rate": [], "val_excess_hit_rate": [], "val_avg_pred_rank": [],
        }
        for topk_val in TOPK_VALUES:
            history[f"train_top{topk_val}_above_mean_rate"] = []
            history[f"val_top{topk_val}_above_mean_rate"] = []
        if has_abtype and prior_topk_abtypes is not None:
            for topk_val in TOPK_VALUES:
                history[f"train_top{topk_val}_vs_prior_rate"] = []
                history[f"val_top{topk_val}_vs_prior_rate"] = []

        best_val_top1_above_mean_rate = -np.inf
        best_state_dict: Optional[dict] = None
        best_epoch: int = 1
        epochs_without_improvement = 0
        has_logged_batch_shape = False

        prior_tensors_on_device: Dict[int, torch.Tensor] = {}
        if has_abtype and prior_topk_abtypes is not None:
            for topk_val, abtype_array in prior_topk_abtypes.items():
                prior_tensors_on_device[topk_val] = torch.tensor(
                    abtype_array, dtype=torch.int64, device=self._device
                )

        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()

            self._net.train()
            epoch_loss_sum = 0.0
            epoch_group_count = 0

            train_group_hit_rates = []
            train_group_excess_rates = []
            train_group_pred_ranks = []
            train_topk_above_mean: Dict[int, list] = {k: [] for k in TOPK_VALUES}
            train_topk_vs_prior: Optional[Dict[int, list]] = (
                {k: [] for k in TOPK_VALUES} if has_abtype else None
            )

            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch:4d}/{num_epochs}",
                leave=False,
                ncols=100,
            )
            for batch in pbar:
                features_padded, labels_padded, labels_original_pad, abtype_padded, valid_mask, group_sizes = batch
                features_padded = features_padded.to(self._device, non_blocking=True)
                labels_padded = labels_padded.to(self._device, non_blocking=True)
                labels_original_pad = labels_original_pad.to(self._device, non_blocking=True)
                abtype_padded = abtype_padded.to(self._device, non_blocking=True)
                valid_mask = valid_mask.to(self._device, non_blocking=True)

                if not has_logged_batch_shape:
                    pbar.write(
                        f"  Batch: X={features_padded.shape}, mask={valid_mask.shape}, "
                        f"max_group={max(group_sizes)}, "
                        f"avg_group={np.mean(group_sizes):.1f}"
                    )
                    has_logged_batch_shape = True

                batch_size_actual = features_padded.shape[0]

                optimizer.zero_grad()
                model_output = self._net(features=features_padded, valid_mask=valid_mask)

                mask_flat = valid_mask.reshape(-1)
                output_flat = model_output.reshape(-1)
                labels_flat = labels_padded.reshape(-1)
                labels_original_flat = labels_original_pad.reshape(-1)

                group_indices = torch.repeat_interleave(
                    torch.arange(batch_size_actual, device=self._device),
                    torch.tensor(group_sizes, device=self._device),
                )

                valid_positions = mask_flat.bool()
                output_valid = output_flat[valid_positions]
                labels_valid = labels_flat[valid_positions]
                labels_original_valid = labels_original_flat[valid_positions]

                target_for_loss = labels_original_valid if use_listmle else labels_valid
                if use_listnet or use_listmle:
                    batch_loss = criterion(
                        predictions=output_valid,
                        targets=target_for_loss,
                        group_indices=group_indices,
                        num_groups=batch_size_actual,
                    )
                else:
                    batch_loss = criterion(predictions=output_valid, targets=target_for_loss)
                batch_loss.backward()

                torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                optimizer.step()

                epoch_loss_sum += batch_loss.item() * batch_size_actual
                epoch_group_count += batch_size_actual

                with torch.no_grad():
                    for batch_idx in range(batch_size_actual):
                        group_len = group_sizes[batch_idx]
                        group_pred = model_output[batch_idx, :group_len]
                        group_labels = labels_padded[batch_idx, :group_len]
                        group_labels_original = labels_original_pad[batch_idx, :group_len]
                        group_abtype = abtype_padded[batch_idx, :group_len]
                        group_mean_label = group_labels_original.mean().item()

                        pred_sort_indices = torch.argsort(group_pred, descending=True)

                        effective_topk = min(topk, group_len)
                        if effective_topk > 0:
                            _, true_topk_indices = torch.topk(group_labels, effective_topk)

                            true_topk_mask = torch.zeros(
                                group_len, dtype=torch.bool, device=self._device
                            )
                            pred_topk_mask = torch.zeros(
                                group_len, dtype=torch.bool, device=self._device
                            )
                            true_topk_mask.scatter_(0, true_topk_indices, True)
                            pred_topk_mask.scatter_(0, pred_sort_indices[:effective_topk], True)

                            hit_count = (true_topk_mask & pred_topk_mask).sum().item()
                            group_hit_rate = hit_count / effective_topk
                            train_group_hit_rates.append(group_hit_rate)

                            baseline_rate = effective_topk / group_len
                            excess_rate = group_hit_rate - baseline_rate
                            train_group_excess_rates.append(excess_rate)

                            pred_top1_index = pred_sort_indices[0].item()
                            true_rank_of_pred_top1 = (
                                torch.argsort(torch.argsort(group_labels_original))[
                                    pred_top1_index
                                ]
                                .item()
                            )
                            normalized_pred_rank = (
                                true_rank_of_pred_top1 / (group_len - 1) if group_len > 1 else 0.0
                            )
                            train_group_pred_ranks.append(normalized_pred_rank)

                        for topk_val in TOPK_VALUES:
                            actual_k = min(topk_val, group_len)
                            if actual_k > 0:
                                pred_topk_indices = pred_sort_indices[:actual_k]
                                pred_topk_mean_label = group_labels_original[pred_topk_indices].mean().item()
                                train_topk_above_mean[topk_val].append(
                                    1.0 if pred_topk_mean_label >= group_mean_label else 0.0
                                )

                                if has_abtype and train_topk_vs_prior is not None:
                                    prior_abtype_tensor = prior_tensors_on_device.get(topk_val)
                                    if prior_abtype_tensor is not None and len(prior_abtype_tensor) > 0:
                                        prior_abtype_mask = torch.isin(group_abtype, prior_abtype_tensor)
                                        if prior_abtype_mask.any():
                                            prior_mean_label = group_labels_original[prior_abtype_mask].mean().item()
                                            train_topk_vs_prior[topk_val].append(
                                                1.0 if pred_topk_mean_label >= prior_mean_label else 0.0
                                            )

            scheduler.step()
            avg_train_loss = epoch_loss_sum / max(epoch_group_count, 1)
            history["train_loss"].append(avg_train_loss)

            train_topk_hit_rate = (
                np.mean(train_group_hit_rates) if train_group_hit_rates else 0.0
            )
            train_excess_hit_rate = (
                np.mean(train_group_excess_rates) if train_group_excess_rates else 0.0
            )
            train_avg_pred_rank = (
                np.mean(train_group_pred_ranks) if train_group_pred_ranks else 0.0
            )
            history["train_topk_hit_rate"].append(train_topk_hit_rate)
            history["train_excess_hit_rate"].append(train_excess_hit_rate)
            history["train_avg_pred_rank"].append(train_avg_pred_rank)

            for topk_val in TOPK_VALUES:
                above_mean_vals = train_topk_above_mean[topk_val]
                history[f"train_top{topk_val}_above_mean_rate"].append(
                    float(np.mean(above_mean_vals)) if above_mean_vals else 0.0
                )
            if has_abtype and train_topk_vs_prior is not None:
                for topk_val in TOPK_VALUES:
                    vs_prior_vals = train_topk_vs_prior[topk_val]
                    history[f"train_top{topk_val}_vs_prior_rate"].append(
                        float(np.mean(vs_prior_vals)) if vs_prior_vals else 0.0
                    )

            val_loss_sum = 0.0
            val_group_count = 0
            val_group_hit_rates = []
            val_group_excess_rates = []
            val_group_pred_ranks = []
            val_topk_above_mean: Dict[int, list] = {k: [] for k in TOPK_VALUES}
            val_topk_vs_prior: Optional[Dict[int, list]] = (
                {k: [] for k in TOPK_VALUES} if has_abtype else None
            )

            if val_loader is not None:
                self._net.eval()
                epoch_val_loss_sum = 0.0

                with torch.no_grad():
                    for batch in val_loader:
                        features_padded, labels_padded, labels_original_pad, abtype_padded, valid_mask, group_sizes = batch
                        features_padded = features_padded.to(self._device, non_blocking=True)
                        labels_padded = labels_padded.to(self._device, non_blocking=True)
                        labels_original_pad = labels_original_pad.to(self._device, non_blocking=True)
                        abtype_padded = abtype_padded.to(self._device, non_blocking=True)
                        valid_mask = valid_mask.to(self._device, non_blocking=True)

                        batch_size_actual = features_padded.shape[0]

                        model_output = self._net(features=features_padded, valid_mask=valid_mask)

                        mask_flat = valid_mask.reshape(-1)
                        output_flat = model_output.reshape(-1)
                        labels_flat = labels_padded.reshape(-1)
                        labels_original_flat = labels_original_pad.reshape(-1)

                        group_indices = torch.repeat_interleave(
                            torch.arange(batch_size_actual, device=self._device),
                            torch.tensor(group_sizes, device=self._device),
                        )

                        valid_positions = mask_flat.bool()
                        output_valid = output_flat[valid_positions]
                        labels_valid = labels_flat[valid_positions]
                        labels_original_valid = labels_original_flat[valid_positions]

                        target_for_loss = labels_original_valid if use_listmle else labels_valid
                        if use_listnet or use_listmle:
                            batch_loss = criterion(
                                predictions=output_valid,
                                targets=target_for_loss,
                                group_indices=group_indices,
                                num_groups=batch_size_actual,
                            )
                        else:
                            batch_loss = criterion(predictions=output_valid, targets=target_for_loss)
                        epoch_val_loss_sum += batch_loss.item() * batch_size_actual
                        val_group_count += batch_size_actual

                        for batch_idx in range(batch_size_actual):
                            group_len = group_sizes[batch_idx]
                            group_pred = model_output[batch_idx, :group_len]
                            group_labels = labels_padded[batch_idx, :group_len]
                            group_labels_original = labels_original_pad[batch_idx, :group_len]
                            group_abtype = abtype_padded[batch_idx, :group_len]
                            group_mean_label = group_labels_original.mean().item()

                            pred_sort_indices = torch.argsort(group_pred, descending=True)

                            effective_topk = min(topk, group_len)
                            if effective_topk > 0:
                                _, true_topk_indices = torch.topk(group_labels, effective_topk)

                                true_topk_mask = torch.zeros(
                                    group_len, dtype=torch.bool, device=self._device
                                )
                                pred_topk_mask = torch.zeros(
                                    group_len, dtype=torch.bool, device=self._device
                                )
                                true_topk_mask.scatter_(0, true_topk_indices, True)
                                pred_topk_mask.scatter_(0, pred_sort_indices[:effective_topk], True)

                                hit_count = (true_topk_mask & pred_topk_mask).sum().item()
                                group_hit_rate = hit_count / effective_topk
                                val_group_hit_rates.append(group_hit_rate)

                                baseline_rate = effective_topk / group_len
                                excess_rate = group_hit_rate - baseline_rate
                                val_group_excess_rates.append(excess_rate)

                                pred_top1_index = pred_sort_indices[0].item()
                                true_rank_of_pred_top1 = (
                                    torch.argsort(torch.argsort(group_labels_original))[
                                        pred_top1_index
                                    ]
                                    .item()
                                )
                                normalized_pred_rank = (
                                    true_rank_of_pred_top1 / (group_len - 1) if group_len > 1 else 0.0
                                )
                                val_group_pred_ranks.append(normalized_pred_rank)

                            for topk_val in TOPK_VALUES:
                                actual_k = min(topk_val, group_len)
                                if actual_k > 0:
                                    pred_topk_indices = pred_sort_indices[:actual_k]
                                    pred_topk_mean_label = group_labels_original[pred_topk_indices].mean().item()
                                    val_topk_above_mean[topk_val].append(
                                        1.0 if pred_topk_mean_label >= group_mean_label else 0.0
                                    )

                                    if has_abtype and val_topk_vs_prior is not None:
                                        prior_abtype_tensor = prior_tensors_on_device.get(topk_val)
                                        if prior_abtype_tensor is not None and len(prior_abtype_tensor) > 0:
                                            prior_abtype_mask = torch.isin(group_abtype, prior_abtype_tensor)
                                            if prior_abtype_mask.any():
                                                prior_mean_label = group_labels_original[
                                                    prior_abtype_mask
                                                ].mean().item()
                                                val_topk_vs_prior[topk_val].append(
                                                    1.0 if pred_topk_mean_label >= prior_mean_label else 0.0
                                                )

                val_loss = epoch_val_loss_sum / max(val_group_count, 1)
                val_topk_hit_rate = (
                    np.mean(val_group_hit_rates) if val_group_hit_rates else 0.0
                )
                val_excess_hit_rate = (
                    np.mean(val_group_excess_rates) if val_group_excess_rates else 0.0
                )
                val_avg_pred_rank = (
                    np.mean(val_group_pred_ranks) if val_group_pred_ranks else 0.0
                )
            else:
                val_topk_hit_rate = 0.0
                val_excess_hit_rate = 0.0
                val_avg_pred_rank = 0.0

            history["val_loss"].append(val_loss)
            history["val_topk_hit_rate"].append(val_topk_hit_rate)
            history["val_excess_hit_rate"].append(val_excess_hit_rate)
            history["val_avg_pred_rank"].append(val_avg_pred_rank)

            for topk_val in TOPK_VALUES:
                above_mean_vals = val_topk_above_mean[topk_val]
                history[f"val_top{topk_val}_above_mean_rate"].append(
                    float(np.mean(above_mean_vals)) if above_mean_vals else 0.0
                )
            if has_abtype and val_topk_vs_prior is not None:
                for topk_val in TOPK_VALUES:
                    vs_prior_vals = val_topk_vs_prior[topk_val]
                    history[f"val_top{topk_val}_vs_prior_rate"].append(
                        float(np.mean(vs_prior_vals)) if vs_prior_vals else 0.0
                    )

            current_val_top1_above_mean = (
                np.mean(val_topk_above_mean[1]) if val_loader and val_topk_above_mean[1] else 0.0
            )
            is_improved = current_val_top1_above_mean - best_val_top1_above_mean_rate >= min_delta
            if is_improved or epoch <= min_epochs:
                if is_improved:
                    best_val_top1_above_mean_rate = current_val_top1_above_mean
                    best_state_dict = {
                        k: v.clone() for k, v in self._net.state_dict().items()
                    }
                    best_epoch = epoch
                    epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            epoch_duration = time.time() - epoch_start_time

            if epoch % 10 == 0 or epoch == 1:
                log_msg = (
                    f"  Epoch {epoch:4d}/{num_epochs} | loss={avg_train_loss:.5f} "
                    f"| val_loss={val_loss:.5f} "
                    f"| train_top{topk}_hit_rate={train_topk_hit_rate:.4f} "
                    f"| train_excess_hit_rate={train_excess_hit_rate:.4f} "
                    f"| train_avg_pred_rank={train_avg_pred_rank:.4f} "
                    f"| val_top{topk}_hit_rate={val_topk_hit_rate:.4f} "
                    f"| val_excess_hit_rate={val_excess_hit_rate:.4f} "
                    f"| val_avg_pred_rank={val_avg_pred_rank:.4f}"
                )
                for topk_val in TOPK_VALUES:
                    metric_key = f"val_top{topk_val}_above_mean_rate"
                    if metric_key in history and history[metric_key]:
                        log_msg += f" | val_top{topk_val}_above_mean={history[metric_key][-1]:.4f}"
                if has_abtype:
                    for topk_val in TOPK_VALUES:
                        metric_key = f"val_top{topk_val}_vs_prior_rate"
                        if metric_key in history and history[metric_key]:
                            log_msg += f" | val_top{topk_val}_vs_prior={history[metric_key][-1]:.4f}"
                log_msg += (
                    (
                        f" | best_val_top1_above_mean={best_val_top1_above_mean_rate:.4f}"
                        if val_loader
                        else ""
                    )
                    + f" | time={epoch_duration:.2f}s"
                )
                pbar.write(log_msg)

            if epoch > min_epochs and val_loader and epochs_without_improvement >= early_stopping_patience:
                print(
                    f"  Early stopping at epoch {epoch} "
                    f"(no improvement for {early_stopping_patience} epochs)."
                )
                break

        if best_state_dict is not None:
            self._net.load_state_dict(best_state_dict)
            if val_loader:
                print(f"  Best model: epoch {best_epoch} | val_top1_above_mean_rate={best_val_top1_above_mean_rate:.4f}")

        self.is_fitted = True
        return history

    def predict(
            self,
            df: pd.DataFrame,
            factor_list: List[str],
            batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Generate predictions for the given DataFrame.

        Args:
            df: DataFrame with feature and grouping columns.
            factor_list: List of feature column names.
            batch_size: Number of groups per prediction batch; defaults to 1024.

        Returns:
            Predicted scores aligned with df rows, shape (N,).

        Raises:
            AssertionError: If the model has not been fitted.
            ValueError: If required grouping columns are missing.
        """
        assert self._net is not None, "Model is not fitted yet."
        self._net.eval()

        required_columns = ["date", "time_sec", "code"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(
                f"DataFrame must have {required_columns} columns for grouped prediction"
            )

        feature_matrix = df[factor_list].fillna(0.0).values.astype(np.float32)
        num_samples = len(df)

        df_work = df.copy().reset_index(drop=True)
        grouped = df_work.groupby(required_columns, sort=False)
        predictions = np.zeros(num_samples, dtype=np.float32)

        all_groups = list(grouped.indices.items())
        if batch_size is None:
            batch_size = 1024

        with torch.no_grad():
            for batch_start in range(0, len(all_groups), batch_size):
                batch_groups = all_groups[batch_start: batch_start + batch_size]

                max_group_len = max(len(row_indices) for _, row_indices in batch_groups)
                num_features = feature_matrix.shape[1]
                batch_size_actual = len(batch_groups)

                features_padded = torch.zeros(batch_size_actual, max_group_len, num_features, dtype=torch.float32)
                valid_mask = torch.zeros(batch_size_actual, max_group_len, dtype=torch.bool)
                group_sizes: List[int] = []

                for batch_idx, (_, group_row_indices) in enumerate(batch_groups):
                    group_len = len(group_row_indices)
                    features_padded[batch_idx, :group_len] = torch.from_numpy(feature_matrix[group_row_indices])
                    valid_mask[batch_idx, :group_len] = True
                    group_sizes.append(group_len)

                features_padded = features_padded.to(self._device)
                valid_mask = valid_mask.to(self._device)

                model_output = self._net(features=features_padded, valid_mask=valid_mask)
                output_numpy = model_output.cpu().numpy()

                for batch_idx, (_, group_row_indices) in enumerate(batch_groups):
                    group_len = group_sizes[batch_idx]
                    predictions[group_row_indices] = output_numpy[batch_idx, :group_len]

        return predictions

    def save(self, path: str) -> None:
        """Save model weights and configuration to disk.

        Args:
            path: File path to save the model checkpoint.
        """
        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True
        )
        torch.save(
            {
                "state_dict": self._net.state_dict(),
                "config": dict(
                    input_dim=self._net.input_proj.in_features,
                    d_model=self.d_model,
                    nhead=self.nhead,
                    num_layers=self.num_layers,
                    dim_ff=self.dim_ff,
                    dropout=self.dropout,
                    apply_batch_softmax=self._net.apply_batch_softmax,
                ),
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load model weights and configuration from disk.

        Args:
            path: File path to load the model checkpoint from.
        """
        checkpoint = torch.load(path, map_location="cpu")
        saved_config = checkpoint["config"]
        self.d_model = saved_config["d_model"]
        self.nhead = saved_config["nhead"]
        self.num_layers = saved_config["num_layers"]
        self.dim_ff = saved_config["dim_ff"]
        self.dropout = saved_config["dropout"]
        self._net = _SetTransformerEncoder(
            input_dim=saved_config["input_dim"],
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_ff=self.dim_ff,
            dropout=self.dropout,
            apply_batch_softmax=saved_config.get("apply_batch_softmax", False),
        )
        self._net.load_state_dict(checkpoint["state_dict"])
        self.is_fitted = True
