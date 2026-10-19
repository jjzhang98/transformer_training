"""PyTorch Dataset wrappers for tabular, grouped, and temporal (sequence) data.

Provides:
    - TabularDataset: Simple pointwise dataset for MLP-like models.
    - GroupedDataset: Group-aware dataset for ListNet/ListMLE losses.
    - TemporalDataset: Sliding-window dataset for sequence models.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .utils import segment_softmax, segment_sum, compute_topk_hit_rate


class TabularDataset(Dataset):
    """Simple pointwise tabular dataset for MLP and other independent-scoring models.

    Optionally applies group-wise z-score normalization on labels if grouping columns
    (date, time_sec, code) are present.

    Args:
        df: DataFrame containing feature and label columns.
        factor_list: List of feature column names (X).
        y_col: Single label column name (y).
    """

    def __init__(self, df: pd.DataFrame, factor_list: List[str], y_col: str) -> None:
        feature_matrix = df[factor_list].values.astype(np.float32)

        group_columns = ["date", "time_sec", "code"]
        has_group_columns = all(col in df.columns for col in group_columns)

        if has_group_columns:

            def zscore_group(group_values):
                if len(group_values) > 1:
                    group_mean = group_values.mean()
                    group_std = max(group_values.std(), 1e-12)
                    return (group_values - group_mean) / group_std
                return np.clip(group_values - group_values.mean(), -3, 3) if len(group_values) == 1 else group_values

            normalized_labels = df.groupby(group_columns, sort=False)[y_col].transform(zscore_group).values.astype(
                np.float32)
        else:
            normalized_labels = df[y_col].values.astype(np.float32)

        self.features = torch.from_numpy(feature_matrix)
        self.labels = torch.from_numpy(normalized_labels)
        self.labels_original = (
            df["_y_original"].values.astype(np.float32) if "_y_original" in df.columns else None
        )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        if self.labels_original is not None:
            return self.features[idx], self.labels[idx], self.labels_original[idx]
        return self.features[idx], self.labels[idx], self.labels[idx]


class GroupedDataset(Dataset):
    """Group-aware dataset that groups samples by (date, time_sec, code).

    Each __getitem__ call returns one group's data. Supports optional
    softmax transformation of labels (for ListNet loss), top-k hard label
    conversion, and label smoothing.

    Args:
        df: DataFrame containing feature, label, and grouping columns.
        factor_list: List of feature column names (X).
        y_col: Single label column name (y).
        apply_softmax_to_y: If True, pre-compute softmax probabilities for labels within each group.
        rank_before_softmax: If True and apply_softmax_to_y is True, rank-transform labels before softmax.
        topk: K value for top-k hard label construction.
        use_hard_label: If True, convert labels to top-k hard labels (1/k for top-k, 0 otherwise);
                        if False, use soft labels (rank + softmax).
        label_smoothing: Label smoothing coefficient in [0, 1]; 0 disables smoothing.
    """

    def __init__(
            self,
            df: pd.DataFrame,
            factor_list: List[str],
            y_col: str,
            apply_softmax_to_y: bool = False,
            rank_before_softmax: bool = True,
            topk: int = 1,
            use_hard_label: bool = False,
            label_smoothing: float = 0.0,
    ) -> None:
        if "date" not in df.columns:
            raise ValueError("GroupedDataset requires 'date' column in DataFrame")
        if "time_sec" not in df.columns:
            raise ValueError("GroupedDataset requires 'time_sec' column in DataFrame")
        if "code" not in df.columns:
            raise ValueError("GroupedDataset requires 'code' column in DataFrame")

        self.dataframe = df
        self.factor_list = factor_list
        self.y_col = y_col
        self.topk = topk
        self.use_hard_label = use_hard_label
        self.label_smoothing = label_smoothing
        self.has_abtype = "abtype" in df.columns

        grouped = df.groupby(["date", "time_sec", "code"], sort=True)
        self.group_keys = list(grouped.indices.keys())
        self.group_indices = grouped.indices

        self.y_softmax_cache = {}
        if apply_softmax_to_y:
            for group_key in self.group_keys:
                sample_indices = self.group_indices[group_key]
                y_values = df[y_col].iloc[sample_indices].values.astype(np.float32)

                if use_hard_label:
                    y_values = self._apply_hard_label(
                        y_values=y_values, topk=topk, smoothing=label_smoothing
                    )
                    y_tensor = torch.from_numpy(y_values)
                    self.y_softmax_cache[group_key] = y_tensor
                else:
                    if rank_before_softmax:
                        y_series = pd.Series(y_values)
                        y_values = y_series.rank(method='average').values.astype(np.float32)
                    y_tensor = torch.from_numpy(y_values)
                    self.y_softmax_cache[group_key] = torch.softmax(y_tensor, dim=0)

    def _apply_hard_label(
            self,
            y_values: np.ndarray,
            topk: int,
            smoothing: float,
    ) -> np.ndarray:
        """Convert raw label values to top-k hard labels with optional label smoothing.

        The top-k items receive equal weight (1/k), ensuring labels sum to 1.
        When smoothing > 0, a uniform background distribution is mixed in.

        Args:
            y_values: Raw label values for one group, shape (group_size,).
            topk: Number of top items to assign positive labels.
            smoothing: Label smoothing coefficient in [0, 1].

        Returns:
            Hard label distribution that sums to 1, shape (group_size,).
        """
        group_size = len(y_values)
        effective_k = min(topk, group_size)

        topk_indices = np.argsort(y_values)[-effective_k:]

        hard_labels = np.zeros(group_size, dtype=np.float32)
        hard_labels[topk_indices] = 1.0 / effective_k

        if smoothing > 0:
            smoothed_labels = np.full(group_size, smoothing / group_size, dtype=np.float32)
            smoothed_labels[topk_indices] = (1 - smoothing) / effective_k + smoothing / group_size
            return smoothed_labels

        return hard_labels

    def __len__(self) -> int:
        return len(self.group_keys)

    def __getitem__(self, idx: int):
        group_key = self.group_keys[idx]
        sample_indices = self.group_indices[group_key]
        group_df = self.dataframe.iloc[sample_indices]

        features = group_df[self.factor_list].to_numpy(dtype=np.float32)
        features = np.nan_to_num(features, nan=0.0)

        labels_original = torch.from_numpy(group_df[self.y_col].to_numpy(dtype=np.float32))

        if group_key in self.y_softmax_cache:
            labels = self.y_softmax_cache[group_key]
        else:
            labels = labels_original

        if self.has_abtype:
            abtype_values = group_df["abtype"].to_numpy(dtype=np.int64).copy()
            abtype_tensor = torch.from_numpy(abtype_values)
        else:
            abtype_tensor = torch.full((len(sample_indices),), -1, dtype=torch.int64)

        return torch.from_numpy(features), labels, labels_original, abtype_tensor


class TemporalDataset(Dataset):
    """Sliding-window dataset for sequence models (Transformer, LSTM, etc.).

    Groups rows by (date, code, abtype), sorts by time_sec within each group,
    then cuts overlapping windows of length ``seq_len`` with step ``stride``.

    Args:
        df: DataFrame (must contain time_sec; optionally date/code/abtype).
        factor_list: Feature column names.
        y_col: Label column name.
        seq_len: Window length T (number of time-steps).
        stride: Step between consecutive window start positions.
                stride=1 → fully overlapping (more data, slower).
                stride=T → non-overlapping (faster, less data).

    Each __getitem__ returns (features_window, labels_window):
        features_window: FloatTensor (T, F)
        labels_window: FloatTensor (T,)
    """

    _GROUP_COLS = ["date", "code", "abtype"]

    def __init__(
            self,
            df: pd.DataFrame,
            factor_list: List[str],
            y_col: str,
            seq_len: int = 20,
            stride: int = 5,
    ) -> None:
        self.seq_len = seq_len
        self.windows: List[tuple] = []

        available_group_cols = [col for col in self._GROUP_COLS if col in df.columns]
        if not available_group_cols:
            self._build_windows(
                group_df=df, factor_list=factor_list, y_col=y_col, seq_len=seq_len, stride=stride
            )
            return

        for _, group_df in df.groupby(available_group_cols, sort=True):
            if "time_sec" in group_df.columns:
                group_df = group_df.sort_values("time_sec")
            self._build_windows(
                group_df=group_df, factor_list=factor_list, y_col=y_col, seq_len=seq_len, stride=stride
            )

    def _build_windows(
            self,
            group_df: pd.DataFrame,
            factor_list: List[str],
            y_col: str,
            seq_len: int,
            stride: int,
    ) -> None:
        """Extract sliding windows from a single group's data.

        Args:
            group_df: DataFrame for one group, sorted by time.
            factor_list: Feature column names.
            y_col: Label column name.
            seq_len: Window length.
            stride: Step between windows.
        """
        group_features = group_df[factor_list].values.astype(np.float32)
        group_labels = group_df[y_col].values.astype(np.float32)
        num_rows = len(group_features)
        for start_pos in range(0, num_rows - seq_len + 1, stride):
            end_pos = start_pos + seq_len
            window_features = group_features[start_pos:end_pos]
            window_labels = group_labels[start_pos:end_pos]
            if not (np.isnan(window_features).any() or np.isnan(window_labels).any()):
                self.windows.append((window_features, window_labels))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        window_features, window_labels = self.windows[idx]
        return torch.from_numpy(window_features), torch.from_numpy(window_labels)
