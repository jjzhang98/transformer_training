"""Multi-Layer Perceptron (MLP) model for tabular financial data.

Architecture:
    Input → [Linear → LayerNorm → ReLU → Dropout] × N → Linear(1)

Training:
    - Pearson/ListNet/ListMLE loss, Adam optimiser
    - Early stopping on validation top1_above_mean_rate
    - Cosine-annealed lr schedule
"""
from __future__ import annotations

import os
import time
import gc
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base import BaseModel
from ..dataset import GroupedDataset
from ..utils import segment_softmax
from ..data import rank_gaussian, resolve_device
from ..losses import get_loss_function, is_loss_compilable
from ..metrics import TOPK_VALUES, build_prior_topk_abtypes


def grouped_collate_fn(batch):
    """Collate function for GroupedDataset that keeps groups separate (no stacking).

    Args:
        batch: List of (features, labels, labels_original, abtype) tuples from GroupedDataset.

    Returns:
        The same list of tuples without any modification.
    """
    return batch


class _MLP(nn.Module):
    """Feed-forward MLP network with optional group-wise softmax output.

    Architecture: Input → [Linear → LayerNorm → ReLU → Dropout] × N → Linear(1)

    When apply_batch_softmax is True, the output is softmax-normalized within
    each group (using segment_softmax) rather than globally.

    Args:
        input_dim: Number of input features.
        hidden_sizes: List of hidden layer dimensions.
        dropout: Dropout probability between layers.
        batch_norm: Whether to apply LayerNorm after each linear layer.
        apply_batch_softmax: Whether to apply group-wise softmax to the output.
    """

    def __init__(
            self,
            input_dim: int,
            hidden_sizes: List[int],
            dropout: float = 0.1,
            batch_norm: bool = True,
            apply_batch_softmax: bool = False,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(current_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)
        self.apply_batch_softmax = apply_batch_softmax

    def forward(
            self,
            features: torch.Tensor,
            group_indices: Optional[torch.Tensor] = None,
            num_groups: Optional[int] = None,
    ) -> torch.Tensor:
        """Forward pass through the MLP.

        Args:
            features: Input feature tensor, shape (N, F).
            group_indices: Group membership index for each sample, shape (N,); required for group-wise softmax.
            num_groups: Total number of distinct groups; required when group_indices is provided.

        Returns:
            Output scores, shape (N,). If apply_batch_softmax is True, scores are
            softmax-normalized within each group.
        """
        raw_scores = self.network(features).squeeze(-1)

        if group_indices is None or not self.apply_batch_softmax:
            if self.apply_batch_softmax:
                raw_scores = torch.softmax(raw_scores, dim=0)
            return raw_scores

        if num_groups is None:
            raise ValueError("num_groups must be provided when group_indices is not None")
        return segment_softmax(logits=raw_scores, group_indices=group_indices, num_groups=num_groups)


class MLPModel(BaseModel):
    """Feed-forward MLP trained with Pearson/ListNet/ListMLE loss.

    Supports group-aware training with early stopping on validation
    top1_above_mean_rate metric.

    Args:
        name: Display name for this model instance.
        hidden_sizes: List of hidden layer dimensions.
        dropout: Dropout probability between layers.
        batch_norm: Whether to apply LayerNorm after each linear layer.
    """

    def __init__(
            self,
            name: str = "mlp",
            hidden_sizes: Optional[List[int]] = None,
            dropout: float = 0.1,
            batch_norm: bool = True,
    ) -> None:
        super().__init__(name)
        self.hidden_sizes = hidden_sizes or [256, 256, 128]
        self.dropout = dropout
        self.batch_norm = batch_norm
        self._net: Optional[_MLP] = None
        self._device: Optional[torch.device] = None

    def fit(
            self,
            df_train: pd.DataFrame,
            factor_list: List[str],
            y_col: str,
            df_val: Optional[pd.DataFrame] = None,
            training_cfg=None,
    ) -> dict:
        """Train the MLP model on grouped data.

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
        batch_size = cfg.batch_size if cfg else 1024
        learning_rate = cfg.learning_rate if cfg else 3e-4
        early_stopping_patience = cfg.early_stopping_patience if cfg else 30
        min_epochs = cfg.min_epochs if cfg else 30
        min_delta = cfg.min_delta if cfg else 1e-4
        device_str = cfg.device if cfg else "auto"
        loss_type = cfg.loss_type if cfg else "mse"
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
        self._net = _MLP(
            input_dim=num_features,
            hidden_sizes=self.hidden_sizes,
            dropout=self.dropout,
            batch_norm=self.batch_norm,
            apply_batch_softmax=use_listnet,
        ).to(self._device, non_blocking=True)

        if use_torch_compile and self._device.type != "cpu":
            self._net = torch.compile(self._net)

        total_params = sum(p.numel() for p in self._net.parameters())
        trainable_params = sum(p.numel() for p in self._net.parameters() if p.requires_grad)
        print(f"\n{'=' * 60}")
        print(f"All Hyperparameters: {self.name} (mlp)")
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
        print(f"  [Model]")
        print(f"    hidden_sizes: {self.hidden_sizes}")
        print(f"    dropout: {self.dropout}")
        print(f"    batch_norm: {self.batch_norm}")
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
            num_workers=0,
            pin_memory=use_pin_memory,
            collate_fn=grouped_collate_fn,
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
                num_workers=0,
                pin_memory=use_pin_memory,
                collate_fn=grouped_collate_fn,
            )

        criterion = get_loss_function(loss_type=loss_type)
        if use_torch_compile and self._device.type != "cpu" and is_loss_compilable(loss_type=loss_type):
            criterion = torch.compile(criterion)

        optimizer = torch.optim.Adam(self._net.parameters(), lr=learning_rate, weight_decay=0)
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

            pbar = tqdm(train_loader, desc=f"Epoch {epoch:4d}/{num_epochs}", leave=False, ncols=100)
            for batch in pbar:
                batch_group_count = len(batch)

                batch_features_list = []
                batch_labels_list = []
                batch_labels_original_list = []
                batch_abtype_list = []
                batch_group_sizes = []

                for group_features, group_labels, group_labels_original, group_abtype in batch:
                    if not has_logged_batch_shape:
                        pbar.write(
                            f"  Group size: X={group_features.shape}, y={group_labels.shape}, "
                            f"X_MB={group_features.element_size() * group_features.nelement() / 1024 / 1024:.2f}, "
                            f"y_MB={group_labels.element_size() * group_labels.nelement() / 1024 / 1024:.2f}"
                        )
                        has_logged_batch_shape = True

                    batch_features_list.append(group_features)
                    batch_labels_list.append(group_labels)
                    batch_labels_original_list.append(group_labels_original)
                    batch_abtype_list.append(group_abtype)
                    batch_group_sizes.append(group_features.shape[0])

                concatenated_features = torch.cat(batch_features_list, dim=0).to(self._device, non_blocking=True)
                concatenated_labels = torch.cat(batch_labels_list, dim=0).to(self._device, non_blocking=True)
                concatenated_labels_original = torch.cat(
                    batch_labels_original_list, dim=0
                ).to(self._device, non_blocking=True)
                concatenated_abtype = torch.cat(batch_abtype_list, dim=0).to(self._device, non_blocking=True)

                group_indices = torch.repeat_interleave(
                    torch.arange(len(batch_group_sizes), device=self._device),
                    torch.tensor(batch_group_sizes, device=self._device),
                )

                optimizer.zero_grad()
                model_output = self._net(
                    features=concatenated_features,
                    group_indices=group_indices,
                    num_groups=len(batch_group_sizes),
                )

                target_for_loss = concatenated_labels_original if use_listmle else concatenated_labels
                if use_listnet or use_listmle:
                    batch_loss = criterion(
                        predictions=model_output,
                        targets=target_for_loss,
                        group_indices=group_indices,
                        num_groups=len(batch_group_sizes),
                    )
                else:
                    batch_loss = criterion(predictions=model_output, targets=target_for_loss)
                batch_loss.backward()

                torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                optimizer.step()

                epoch_loss_sum += batch_loss.item() * batch_group_count
                epoch_group_count += batch_group_count

                with torch.no_grad():
                    pred_per_group = torch.split(model_output, batch_group_sizes)
                    labels_per_group = torch.split(concatenated_labels, batch_group_sizes)
                    labels_original_per_group = torch.split(concatenated_labels_original, batch_group_sizes)
                    abtype_per_group = torch.split(concatenated_abtype, batch_group_sizes)

                    for group_pred, group_labels, group_labels_original, group_abtype, group_size in zip(
                        pred_per_group, labels_per_group, labels_original_per_group,
                        abtype_per_group, batch_group_sizes,
                    ):
                        group_mean_label = group_labels_original.mean().item()

                        pred_sort_indices = torch.argsort(group_pred, descending=True)

                        effective_topk = min(topk, group_size)
                        if effective_topk > 0:
                            _, true_topk_indices = torch.topk(group_labels, effective_topk)

                            true_topk_mask = torch.zeros(
                                group_size, dtype=torch.bool, device=self._device
                            )
                            pred_topk_mask = torch.zeros(
                                group_size, dtype=torch.bool, device=self._device
                            )
                            true_topk_mask.scatter_(0, true_topk_indices, True)
                            pred_topk_mask.scatter_(0, pred_sort_indices[:effective_topk], True)

                            hit_count = (true_topk_mask & pred_topk_mask).sum().item()
                            group_hit_rate = hit_count / effective_topk
                            train_group_hit_rates.append(group_hit_rate)

                            baseline_rate = effective_topk / group_size
                            excess_rate = group_hit_rate - baseline_rate
                            train_group_excess_rates.append(excess_rate)

                            pred_top1_index = pred_sort_indices[0].item()
                            true_rank_of_pred_top1 = torch.argsort(
                                torch.argsort(group_labels_original)
                            )[pred_top1_index].item()
                            normalized_pred_rank = (
                                true_rank_of_pred_top1 / (group_size - 1) if group_size > 1 else 0.0
                            )
                            train_group_pred_ranks.append(normalized_pred_rank)

                        for topk_val in TOPK_VALUES:
                            actual_k = min(topk_val, group_size)
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
            avg_train_loss = epoch_loss_sum / epoch_group_count
            history["train_loss"].append(avg_train_loss)

            train_topk_hit_rate = np.mean(train_group_hit_rates) if train_group_hit_rates else 0.0
            train_excess_hit_rate = np.mean(train_group_excess_rates) if train_group_excess_rates else 0.0
            train_avg_pred_rank = np.mean(train_group_pred_ranks) if train_group_pred_ranks else 0.0
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
                        batch_group_count = len(batch)

                        batch_features_list = []
                        batch_labels_list = []
                        batch_labels_original_list = []
                        batch_abtype_list = []
                        batch_group_sizes = []

                        for group_features, group_labels, group_labels_original, group_abtype in batch:
                            batch_features_list.append(group_features)
                            batch_labels_list.append(group_labels)
                            batch_labels_original_list.append(group_labels_original)
                            batch_abtype_list.append(group_abtype)
                            batch_group_sizes.append(group_features.shape[0])

                        concatenated_features = torch.cat(
                            batch_features_list, dim=0
                        ).to(self._device, non_blocking=True)
                        concatenated_labels = torch.cat(
                            batch_labels_list, dim=0
                        ).to(self._device, non_blocking=True)
                        concatenated_labels_original = torch.cat(
                            batch_labels_original_list, dim=0
                        ).to(self._device, non_blocking=True)
                        concatenated_abtype = torch.cat(
                            batch_abtype_list, dim=0
                        ).to(self._device, non_blocking=True)

                        group_indices = torch.repeat_interleave(
                            torch.arange(len(batch_group_sizes), device=self._device),
                            torch.tensor(batch_group_sizes, device=self._device),
                        )

                        model_output = self._net(
                            features=concatenated_features,
                            group_indices=group_indices,
                            num_groups=len(batch_group_sizes),
                        )

                        target_for_loss = concatenated_labels_original if use_listmle else concatenated_labels
                        if use_listnet or use_listmle:
                            batch_loss = criterion(
                                predictions=model_output,
                                targets=target_for_loss,
                                group_indices=group_indices,
                                num_groups=len(batch_group_sizes),
                            )
                        else:
                            batch_loss = criterion(predictions=model_output, targets=target_for_loss)
                        epoch_val_loss_sum += batch_loss.item() * batch_group_count
                        val_group_count += batch_group_count

                        pred_per_group = torch.split(model_output, batch_group_sizes)
                        labels_per_group = torch.split(concatenated_labels, batch_group_sizes)
                        labels_original_per_group = torch.split(concatenated_labels_original, batch_group_sizes)
                        abtype_per_group = torch.split(concatenated_abtype, batch_group_sizes)

                        for group_pred, group_labels, group_labels_original, group_abtype, group_size in zip(
                            pred_per_group, labels_per_group, labels_original_per_group,
                            abtype_per_group, batch_group_sizes,
                        ):
                            group_mean_label = group_labels_original.mean().item()

                            pred_sort_indices = torch.argsort(group_pred, descending=True)

                            effective_topk = min(topk, group_size)
                            if effective_topk > 0:
                                _, true_topk_indices = torch.topk(group_labels, effective_topk)

                                true_topk_mask = torch.zeros(
                                    group_size, dtype=torch.bool, device=self._device
                                )
                                pred_topk_mask = torch.zeros(
                                    group_size, dtype=torch.bool, device=self._device
                                )
                                true_topk_mask.scatter_(0, true_topk_indices, True)
                                pred_topk_mask.scatter_(0, pred_sort_indices[:effective_topk], True)

                                hit_count = (true_topk_mask & pred_topk_mask).sum().item()
                                group_hit_rate = hit_count / effective_topk
                                val_group_hit_rates.append(group_hit_rate)

                                baseline_rate = effective_topk / group_size
                                excess_rate = group_hit_rate - baseline_rate
                                val_group_excess_rates.append(excess_rate)

                                pred_top1_index = pred_sort_indices[0].item()
                                true_rank_of_pred_top1 = torch.argsort(
                                    torch.argsort(group_labels_original)
                                )[pred_top1_index].item()
                                normalized_pred_rank = (
                                    true_rank_of_pred_top1 / (group_size - 1) if group_size > 1 else 0.0
                                )
                                val_group_pred_ranks.append(normalized_pred_rank)

                            for topk_val in TOPK_VALUES:
                                actual_k = min(topk_val, group_size)
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

                val_loss = epoch_val_loss_sum / val_group_count

                val_topk_hit_rate = np.mean(val_group_hit_rates) if val_group_hit_rates else 0.0
                val_excess_hit_rate = np.mean(val_group_excess_rates) if val_group_excess_rates else 0.0
                val_avg_pred_rank = np.mean(val_group_pred_ranks) if val_group_pred_ranks else 0.0
            else:
                val_loss = 0.0
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
            is_improved = (current_val_top1_above_mean - best_val_top1_above_mean_rate >= min_delta)

            if is_improved or epoch <= min_epochs:
                if is_improved:
                    best_val_top1_above_mean_rate = current_val_top1_above_mean
                    best_state_dict = {k: v.clone() for k, v in self._net.state_dict().items()}
                    best_epoch = epoch
                    epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            epoch_duration = time.time() - epoch_start_time

            if epoch % 10 == 0 or epoch == 1:
                log_msg = (
                    f"  Epoch {epoch:4d}/{num_epochs} | loss={avg_train_loss:.5f} | val_loss={val_loss:.5f} "
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
                    (f" | best_val_top1_above_mean={best_val_top1_above_mean_rate:.4f}" if val_loader else "")
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

        feature_matrix = df[factor_list].fillna(0.0).values.astype(np.float32)
        num_samples, num_features = feature_matrix.shape

        if "date" in df.columns and "time_sec" in df.columns and "code" in df.columns:
            df_work = df.copy().reset_index(drop=True)
            grouped = df_work.groupby(["date", "time_sec", "code"], sort=False)
            predictions = np.zeros(num_samples, dtype=np.float32)

            all_groups = list(grouped.indices.items())

            if batch_size is None:
                batch_size = 1024

            with torch.no_grad():
                for batch_start in range(0, len(all_groups), batch_size):
                    batch_end = min(batch_start + batch_size, len(all_groups))
                    batch_groups = all_groups[batch_start:batch_end]

                    batch_features_list = []
                    batch_group_sizes = []

                    for _, group_row_indices in batch_groups:
                        group_features = feature_matrix[group_row_indices]
                        batch_features_list.append(group_features)
                        batch_group_sizes.append(len(group_row_indices))

                    concatenated_features = torch.tensor(
                        np.vstack(batch_features_list), device=self._device, dtype=torch.float32
                    )

                    num_groups_in_batch = len(batch_groups)
                    group_indices = torch.repeat_interleave(
                        torch.arange(num_groups_in_batch, device=self._device),
                        torch.tensor(batch_group_sizes, device=self._device),
                    )

                    model_output = self._net(
                        features=concatenated_features,
                        group_indices=group_indices,
                        num_groups=num_groups_in_batch,
                    )

                    output_numpy = model_output.cpu().numpy()
                    offset = 0
                    for (_, group_row_indices) in batch_groups:
                        group_length = len(group_row_indices)
                        predictions[group_row_indices] = output_numpy[offset:offset + group_length]
                        offset += group_length

            return predictions
        else:
            raise ValueError("DataFrame must have 'date', 'time_sec', 'code' columns for grouped prediction")

    def save(self, path: str) -> None:
        """Save model weights to disk.

        Args:
            path: File path to save the model state dict.
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(self._net.state_dict(), path)

    def load(self, path: str) -> None:
        """Load model weights from disk.

        Args:
            path: File path to load the model state dict from.

        Raises:
            AssertionError: If the network has not been initialized.
        """
        assert self._net is not None, "Build the network first (call fit or set architecture)."
        self._net.load_state_dict(torch.load(path, map_location="cpu"))
        self.is_fitted = True
