"""LambdaMART model for Learning-to-Rank using LightGBM.

Implements the LambdaMART algorithm which combines MART (Multiple Additive
Regression Trees) with LambdaRank's gradient scaling. This module overrides
LightGBM's default objective and evaluation functions with custom ranking
metrics (topk_hit_rate, excess_hit_rate, avg_pred_rank, topk_above_mean_rate).
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from .base import BaseModel
from ..metrics import TOPK_VALUES, build_prior_topk_abtypes


def _make_custom_early_stopping(
        stopping_rounds: int,
        metric_name: str,
        verbose: bool,
        valid_name: str = "val",
):
    """Create a custom early stopping callback for LightGBM training.

    Monitors a specific metric on a specific validation set and stops
    training when no improvement is observed for `stopping_rounds` iterations.

    Args:
        stopping_rounds: Number of iterations without improvement before stopping.
        metric_name: Name of the metric to monitor (e.g. "top1_above_mean").
        verbose: Whether to print early stopping progress.
        valid_name: Name of the validation set to monitor (e.g. "val").

    Returns:
        A callback function compatible with LightGBM's callback system.
    """
    target_metric = metric_name
    best_score = -np.inf
    best_iteration = 0
    no_improve_count = 0
    has_warned_missing = False

    def _callback(env):
        nonlocal best_score, best_iteration, no_improve_count, has_warned_missing
        if not env.evaluation_result_list:
            return
        metric_found = False
        for data_name, metric, score, is_higher_better in env.evaluation_result_list:
            if data_name == valid_name and metric == target_metric:
                metric_found = True
                if is_higher_better:
                    is_improved = score >= best_score + 1e-6
                else:
                    is_improved = score <= best_score - 1e-6
                if is_improved:
                    best_score = score
                    best_iteration = env.iteration
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                if verbose and (env.iteration % 10 == 0 or env.iteration <= 1):
                    direction = "↑" if is_higher_better else "↓"
                    print(
                        f"  [EarlyStopping] iter={env.iteration} | "
                        f"{valid_name}_{target_metric}={score:.4f}{direction} | "
                        f"best={best_score:.4f} | "
                        f"no_improve={no_improve_count}/{stopping_rounds}"
                    )
                if no_improve_count >= stopping_rounds:
                    if verbose:
                        print(
                            f"  Early stopping at iteration {env.iteration} "
                            f"(best: {best_score:.4f} at iter {best_iteration})"
                        )
                    raise lgb.callback.EarlyStopException(
                        best_iteration, env.evaluation_result_list
                    )
                return
        if not metric_found and not has_warned_missing:
            has_warned_missing = True
            available_metrics = [
                (data_name, metric, "H" if is_higher else "L")
                for data_name, metric, _, is_higher in env.evaluation_result_list
            ]
            import warnings
            warnings.warn(
                f"EarlyStopping: '{target_metric}' not found in "
                f"'{valid_name}' set. Available metrics: {available_metrics}. "
                f"Early stopping is effectively DISABLED.",
                RuntimeWarning,
            )

    _callback.order = 30
    return _callback


_LAMBDAMART_DEFAULTS = dict(
    objective="lambdarank",
    boosting_type="gbdt",
    num_leaves=63,
    learning_rate=0.05,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    lambda_l1=0.1,
    lambda_l2=0.1,
    min_data_in_leaf=20,
    verbose=-1,
    num_threads=-1,
    lambdarank_truncation_level=30,
)


class LambdaMARTModel(BaseModel):
    """LambdaMART ranking model built on LightGBM.

    Overrides LightGBM's default evaluation with custom ranking metrics
    (topk_hit_rate, excess_hit_rate, avg_pred_rank, topk_above_mean_rate,
    topk_vs_prior_rate) and uses custom early stopping on top1_above_mean.

    Args:
        name: Display name for this model instance.
        **params: Additional LightGBM parameters; overrides _LAMBDAMART_DEFAULTS.
                  Special keys: early_stopping_patience (default 50, now read from training_cfg).
    """

    def __init__(self, name: str = "lambdamart", **params) -> None:
        super().__init__(name)
        self.params = {**_LAMBDAMART_DEFAULTS, **params}
        self._model: Optional[lgb.Booster] = None
        self._feature_names: Optional[List[str]] = None

    @staticmethod
    def _prepare_grouped_data(
            df: pd.DataFrame,
            factor_list: List[str],
            y_col: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Sort and prepare data for LightGBM's group-based training.

        Groups by (date, time_sec, code), computes dense ranks for labels,
        and extracts group sizes and optional abtype information.

        Args:
            df: DataFrame with feature, label, and grouping columns.
            factor_list: List of feature column names.
            y_col: Label column name.

        Returns:
            Tuple of (features, raw_labels, ranked_labels, group_sizes, abtype_array).
        """
        group_columns = ["date", "time_sec", "code"]
        df_sorted = df.sort_values(group_columns).reset_index(drop=True)
        features = df_sorted[factor_list].values.astype(np.float32)
        raw_labels = df_sorted[y_col].values.astype(np.float32)
        ranked_labels = (
            df_sorted.groupby(group_columns, sort=False)[y_col]
            .transform(lambda group: group.rank(method="dense"))
            .values.astype(np.int32)
        )
        group_sizes = (
            df_sorted.groupby(group_columns, sort=False).size().values.astype(np.int32)
        )
        abtype_array = (
            df_sorted["abtype"].values.astype(np.int64)
            if "abtype" in df_sorted.columns
            else None
        )
        return features, raw_labels, ranked_labels, group_sizes, abtype_array

    @staticmethod
    def _compute_ranking_metrics(
            predictions: np.ndarray,
            raw_labels: np.ndarray,
            group_sizes: np.ndarray,
            topk: int,
            abtype: Optional[np.ndarray] = None,
            prior_topk_abtypes: Optional[Dict[int, np.ndarray]] = None,
    ) -> Dict[str, float]:
        """Compute all ranking metrics using vectorized 2D array operations.

        Groups are packed into a 2D array (n_groups, max_size) with -inf padding,
        then batch operations (argsort, cumsum, broadcasted intersect1d) are applied.

        Key optimizations:
        1. Batch argsort via np.argsort(-pred_2d, axis=1)
        2. Batch cumsum for top-k means
        3. Broadcasted intersect1d: (n,k,1) == (n,1,k) -> count matches
        4. Vectorized rank calculation via argsort(argsort(...))

        Args:
            predictions: Model prediction scores, shape (N,).
            raw_labels: Ground-truth label values, shape (N,).
            group_sizes: Number of samples per group, shape (n_groups,).
            topk: K value for top-k hit rate calculation.
            abtype: Optional abtype values per sample, shape (N,).
            prior_topk_abtypes: Optional dict mapping k -> array of prior top-k abtype values.

        Returns:
            Dictionary of metric names to float values.
        """
        num_groups = len(group_sizes)
        if num_groups == 0:
            return {}
        group_offsets = np.concatenate([[0], np.cumsum(group_sizes)])
        max_group_size = int(group_sizes.max())
        max_topk = max(TOPK_VALUES)
        has_prior = abtype is not None and prior_topk_abtypes is not None

        pred_2d = np.full((num_groups, max_group_size), -np.inf, dtype=predictions.dtype)
        true_2d = np.full((num_groups, max_group_size), -np.inf, dtype=raw_labels.dtype)
        valid_mask = np.zeros((num_groups, max_group_size), dtype=bool)

        for group_idx in range(num_groups):
            start_offset = group_offsets[group_idx]
            end_offset = group_offsets[group_idx + 1]
            group_len = int(group_sizes[group_idx])
            pred_2d[group_idx, :group_len] = predictions[start_offset:end_offset]
            true_2d[group_idx, :group_len] = raw_labels[start_offset:end_offset]
            valid_mask[group_idx, :group_len] = True

        group_mean_labels = np.zeros(num_groups, dtype=raw_labels.dtype)
        for group_idx in range(num_groups):
            group_mean_labels[group_idx] = np.mean(
                raw_labels[group_offsets[group_idx]:group_offsets[group_idx + 1]]
            )

        pred_sort_indices_2d = np.argsort(-pred_2d, axis=1)
        true_sort_indices_2d = np.argsort(-true_2d, axis=1)

        effective_k = topk
        true_topk_indices_2d = true_sort_indices_2d[:, :effective_k]
        pred_topk_indices_2d = pred_sort_indices_2d[:, :effective_k]

        if effective_k == 1:
            hit_counts = (true_topk_indices_2d[:, 0] == pred_topk_indices_2d[:, 0]).astype(float)
        else:
            match_matrix = (pred_topk_indices_2d[:, :, None] == true_topk_indices_2d[:, None, :])
            hit_counts = match_matrix.any(axis=2).sum(axis=1).astype(float)

        hit_rates = hit_counts / effective_k
        excess_rates = hit_rates - effective_k / group_sizes

        true_rank_descending = np.argsort(true_sort_indices_2d, axis=1)
        row_indices = np.arange(num_groups)
        rank_of_pred_top1_desc = true_rank_descending[row_indices, pred_topk_indices_2d[:, 0]]
        true_rank_ascending = group_sizes - 1 - rank_of_pred_top1_desc
        pred_ranks = np.where(
            group_sizes > 1,
            true_rank_ascending.astype(np.float64) / (group_sizes - 1),
            0.0,
        )

        row_indices_for_topk = np.arange(num_groups)[:, None]
        topk_sort_indices = pred_sort_indices_2d[:, :max_topk]
        topk_true_values = true_2d[row_indices_for_topk, topk_sort_indices]
        topk_true_values = np.where(np.isfinite(topk_true_values), topk_true_values, 0)
        cumsum_values = np.cumsum(topk_true_values, axis=1)
        topk_mean_2d = cumsum_values / np.arange(1, max_topk + 1)

        topk_above_mean: Dict[int, np.ndarray] = {}
        topk_vs_prior_results: Dict[int, np.ndarray] = {}

        for topk_val in TOPK_VALUES:
            if topk_val <= max_group_size:
                is_above_mean = topk_mean_2d[:, topk_val - 1] >= group_mean_labels
                group_valid_for_topk = group_sizes >= topk_val
                topk_above_mean[topk_val] = is_above_mean[group_valid_for_topk]
                if has_prior and abtype is not None:
                    prior_abtypes_for_k = prior_topk_abtypes.get(topk_val, np.array([]))
                    if len(prior_abtypes_for_k) > 0:
                        vs_prior_list = []
                        for group_idx in range(num_groups):
                            if group_sizes[group_idx] < topk_val:
                                continue
                            start_offset = group_offsets[group_idx]
                            end_offset = group_offsets[group_idx + 1]
                            group_abtype = abtype[start_offset:end_offset]
                            group_true = raw_labels[start_offset:end_offset]
                            prior_abtype_mask = np.isin(group_abtype, prior_abtypes_for_k)
                            if prior_abtype_mask.any():
                                prior_mean = np.mean(group_true[prior_abtype_mask])
                                is_above_prior = 1.0 if topk_mean_2d[group_idx, topk_val - 1] >= prior_mean else 0.0
                                vs_prior_list.append(is_above_prior)
                        if vs_prior_list:
                            topk_vs_prior_results[topk_val] = np.array(vs_prior_list)

        result = {
            "topk_hit_rate": float(np.mean(hit_rates)),
            "excess_hit_rate": float(np.mean(excess_rates)),
            "avg_pred_rank": float(np.mean(pred_ranks)),
        }
        for topk_val in TOPK_VALUES:
            above_mean_vals = topk_above_mean.get(topk_val)
            if above_mean_vals is not None and len(above_mean_vals) > 0:
                result[f"top{topk_val}_above_mean_rate"] = float(np.mean(above_mean_vals.astype(float)))
        if has_prior:
            for topk_val in TOPK_VALUES:
                vs_prior_vals = topk_vs_prior_results.get(topk_val)
                if vs_prior_vals is not None and len(vs_prior_vals) > 0:
                    result[f"top{topk_val}_vs_prior_rate"] = float(np.mean(vs_prior_vals.astype(float)))
        return result

    def fit(
            self,
            df_train: pd.DataFrame,
            factor_list: List[str],
            y_col: str,
            df_val: Optional[pd.DataFrame] = None,
            training_cfg=None,
    ) -> dict:
        """Train the LambdaMART model using LightGBM.

        Args:
            df_train: Training DataFrame with feature, label, and grouping columns.
            factor_list: List of feature column names.
            y_col: Label column name.
            df_val: Optional validation DataFrame for early stopping.
            training_cfg: TrainingConfig dataclass with hyperparameters.

        Returns:
            History dict with per-iteration metrics for train and validation.
        """
        cfg = training_cfg
        topk = cfg.topk if cfg else 1
        num_boost_round = cfg.epochs if cfg else 2000
        early_stopping_patience = cfg.early_stopping_patience if cfg else 50
        self._feature_names = factor_list
        train_features, train_raw_labels, train_ranked_labels, train_group_sizes, train_abtype = (
            self._prepare_grouped_data(df=df_train, factor_list=factor_list, y_col=y_col)
        )
        prior_topk_abtypes = None
        if "abtype" in df_train.columns:
            df_union = (
                pd.concat([df_train, df_val], ignore_index=True)
                if df_val is not None and not df_val.empty
                else df_train
            )
            prior_topk_abtypes = build_prior_topk_abtypes(df=df_union, y_col=y_col)
        lgb_train = lgb.Dataset(
            data=train_features,
            label=train_ranked_labels,
            feature_name=factor_list,
            group=train_group_sizes,
            free_raw_data=True,
        )
        valid_sets = [lgb_train]
        valid_names = ["train"]
        eval_data: Dict[int, Tuple] = {
            id(lgb_train): (train_raw_labels, train_group_sizes, "train", train_abtype)
        }
        lgb_val = None
        if df_val is not None and not df_val.empty:
            val_features, val_raw_labels, val_ranked_labels, val_group_sizes, val_abtype = (
                self._prepare_grouped_data(df=df_val, factor_list=factor_list, y_col=y_col)
            )
            lgb_val = lgb.Dataset(
                data=val_features,
                label=val_ranked_labels,
                feature_name=factor_list,
                group=val_group_sizes,
                reference=lgb_train,
            )
            valid_sets.append(lgb_val)
            valid_names.append("val")
            eval_data[id(lgb_val)] = (val_raw_labels, val_group_sizes, "val", val_abtype)

        def _ranking_eval_fn(y_pred, dataset):
            """Custom evaluation function for LightGBM that computes ranking metrics."""
            dataset_id = id(dataset)
            if dataset_id not in eval_data:
                return [("unknown", 0.0, True)]
            raw_labels, group_sizes, _, abtype_array = eval_data[dataset_id]
            metrics = LambdaMARTModel._compute_ranking_metrics(
                predictions=y_pred,
                raw_labels=raw_labels,
                group_sizes=group_sizes,
                topk=topk,
                abtype=abtype_array,
                prior_topk_abtypes=prior_topk_abtypes,
            )
            results = [
                ("topk_hit_rate", metrics["topk_hit_rate"], True),
                ("excess_hit_rate", metrics["excess_hit_rate"], True),
                ("avg_pred_rank", metrics["avg_pred_rank"], True),
            ]
            for topk_val in TOPK_VALUES:
                metric_key = f"top{topk_val}_above_mean_rate"
                if metric_key in metrics:
                    results.append((f"top{topk_val}_above_mean", metrics[metric_key], True))
            for topk_val in TOPK_VALUES:
                metric_key = f"top{topk_val}_vs_prior_rate"
                if metric_key in metrics:
                    results.append((f"top{topk_val}_vs_prior", metrics[metric_key], True))
            return results

        print(f"\n{'=' * 60}")
        print(f"LambdaMART Model Configuration:")
        print(f"{'=' * 60}")
        print(f"  Objective: lambdarank")
        print(f"  num_boost_round (epochs): {num_boost_round}")
        print(f"  early_stopping_patience: {early_stopping_patience}")
        print(f"  topk: {topk}")
        print(f"  Training groups: {len(train_group_sizes)}")
        print(f"  Training samples: {len(train_features)}")
        if lgb_val is not None:
            print(f"  Validation groups: {len(val_group_sizes)}")
            print(f"  Validation samples: {len(val_features)}")
        print(f"  Params: {self.params}")
        print(f"{'=' * 60}\n")

        evals_result: dict = {}
        callbacks = [
            lgb.log_evaluation(period=50),
            lgb.record_evaluation(evals_result),
        ]
        early_stop_rounds = early_stopping_patience
        if df_val is None or df_val.empty:
            early_stop_rounds = None
        if early_stop_rounds is not None:
            callbacks.append(
                _make_custom_early_stopping(
                    stopping_rounds=early_stop_rounds,
                    metric_name="top1_above_mean",
                    verbose=True,
                )
            )

        self._model = lgb.train(
            params=self.params,
            train_set=lgb_train,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            feval=_ranking_eval_fn,
            callbacks=callbacks,
        )
        self.is_fitted = True

        best_iteration = self._model.best_iteration
        if best_iteration is not None and best_iteration > 0 and lgb_val is not None:
            print(f"  Best iteration: {best_iteration} trees")

        history: dict = {}
        for split_name in ["train", "val"]:
            if split_name not in evals_result:
                continue
            for eval_key, values in evals_result[split_name].items():
                if eval_key == "topk_hit_rate":
                    history[f"{split_name}_topk_hit_rate"] = values
                elif eval_key == "excess_hit_rate":
                    history[f"{split_name}_excess_hit_rate"] = values
                elif eval_key == "avg_pred_rank":
                    history[f"{split_name}_avg_pred_rank"] = values
                elif eval_key == "top1_above_mean":
                    history[f"{split_name}_top1_above_mean_rate"] = values
                elif eval_key.startswith("top") and eval_key.endswith("_above_mean"):
                    topk_str = eval_key.replace("top", "").replace("_above_mean", "")
                    history[f"{split_name}_top{topk_str}_above_mean_rate"] = values
                elif eval_key.startswith("top") and eval_key.endswith("_vs_prior"):
                    topk_str = eval_key.replace("top", "").replace("_vs_prior", "")
                    history[f"{split_name}_top{topk_str}_vs_prior_rate"] = values
                elif "ndcg" in eval_key.lower():
                    loss_key = f"{split_name}_loss"
                    if loss_key not in history:
                        history[loss_key] = [1.0 - v for v in values]
        return history

    def predict(self, df: pd.DataFrame, factor_list: List[str]) -> np.ndarray:
        """Generate predictions for the given DataFrame.

        Args:
            df: DataFrame with feature columns.
            factor_list: List of feature column names.

        Returns:
            Predicted scores aligned with df rows, shape (N,).

        Raises:
            AssertionError: If the model has not been fitted.
        """
        assert self._model is not None, "Model is not fitted yet."
        features = df[factor_list].fillna(0.0).values.astype(np.float32)
        return self._model.predict(features, num_iteration=self._model.best_iteration)

    def feature_importance(self) -> Optional[pd.Series]:
        """Return feature importance scores (gain-based).

        Returns:
            pd.Series of gain-based feature importances, or None if model is not fitted.
        """
        if self._model is None or self._feature_names is None:
            return None
        gain_scores = self._model.feature_importance(importance_type="gain")
        return pd.Series(gain_scores, index=self._feature_names, name="gain")

    def save(self, path: str) -> None:
        """Save the LightGBM model to disk.

        Args:
            path: File path to save the model.
        """
        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True
        )
        self._model.save_model(path, num_iteration=self._model.best_iteration)

    def load(self, path: str) -> None:
        """Load the LightGBM model from disk.

        Args:
            path: File path to load the model from.
        """
        self._model = lgb.Booster(model_file=path)
        self.is_fitted = True
