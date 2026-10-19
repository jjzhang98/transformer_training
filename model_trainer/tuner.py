"""Hyperparameter tuning module for automated model optimization.

Supports three search strategies:
    - Random search (default): Samples hyperparameters uniformly at random.
    - Grid search: Exhaustively evaluates all combinations.
    - Optuna Bayesian optimization: Uses TPE sampler for intelligent search.

The optimization objective is the early-stopping metric used by each model
(val_top1_above_mean_rate by default), ensuring that the selected
hyperparameters maximize the same criterion that governs training termination.

Usage:
    from model_trainer.tuner import HyperparamTuner

    tuner = HyperparamTuner(
        base_config=cfg,
        method="random",          # "random" | "grid" | "optuna"
        n_trials=50,
    )
    result = tuner.run()
    best_cfg = result["best_config"]
"""
from __future__ import annotations

import copy
import gc
import itertools
import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .config import Config, TrainingConfig
from .data import load_all_splits
from .models import build_model

_DEFAULT_METRICS: Dict[str, str] = {
    "mlp": "val_top1_above_mean_rate",
    "transformer": "val_top1_above_mean_rate",
    "lambdamart": "val_top1_above_mean_rate",
    "gru": "val_ic",
    "dlinear": "val_ic",
    "lgbm": "val_ic",
    "lasso": "val_ic",
}

_DEFAULT_SEARCH_SPACES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "mlp": {
        "training": {
            "learning_rate": {"type": "float", "low": 1e-6, "high": 1e-4, "log": True},
            "batch_size": {"type": "categorical", "choices": [2048, 4096, 8192]},
        },
        "model_params": {
            "hidden_sizes": {
                "type": "categorical",
                "choices": [[256, 128, 64], [512, 256, 128], [256, 256, 128], [512, 256, 64]],
            },
            "dropout": {"type": "float", "low": 0.0, "high": 0.5},
        },
    },
    "transformer": {
        "training": {
            "learning_rate": {"type": "float", "low": 1e-6, "high": 1e-4, "log": True},
            "batch_size": {"type": "categorical", "choices": [256, 512, 1024]},
        },
        "model_params": {
            "d_model": {"type": "categorical", "choices": [128, 256, 512]},
            "nhead": {"type": "categorical", "choices": [4, 8]},
            "num_layers": {"type": "int", "low": 1, "high": 4},
            "dim_ff": {"type": "categorical", "choices": [512, 1024, 2048]},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5},
        },
    },
    "lambdamart": {
        "training": {},
        "model_params": {
            "num_leaves": {"type": "int", "low": 1, "high": 95, "step": 4},
            "learning_rate": {"type": "float", "low": 0.005, "high": 0.1, "log": True},
            "feature_fraction": {"type": "float", "low": 0.5, "high": 1.0},
            "bagging_fraction": {"type": "float", "low": 0.5, "high": 1.0},
            "lambda_l1": {"type": "float", "low": 0.0, "high": 1.0},
            "lambda_l2": {"type": "float", "low": 0.0, "high": 1.0},
            "min_data_in_leaf": {"type": "int", "low": 5, "high": 100, "step": 5},
            "bagging_freq": {"type": "int", "low": 1, "high": 10, "step": 1},
        },
    },
    "gru": {
        "training": {
            "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
            "batch_size": {"type": "categorical", "choices": [256, 512, 1024]},
        },
        "model_params": {
            "hidden_size": {"type": "categorical", "choices": [64, 128, 256]},
            "num_layers": {"type": "int", "low": 1, "high": 4},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5},
        },
    },
    "dlinear": {
        "training": {
            "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
            "batch_size": {"type": "categorical", "choices": [256, 512, 1024]},
        },
        "model_params": {
            "hidden_size": {"type": "categorical", "choices": [64, 128, 256, 512]},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5},
        },
    },
    "lgbm": {
        "training": {},
        "model_params": {
            "num_leaves": {"type": "int", "low": 15, "high": 127, "step": 4},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "feature_fraction": {"type": "float", "low": 0.5, "high": 1.0},
            "bagging_fraction": {"type": "float", "low": 0.5, "high": 1.0},
            "lambda_l1": {"type": "float", "low": 0.0, "high": 1.0},
            "lambda_l2": {"type": "float", "low": 0.0, "high": 1.0},
            "min_data_in_leaf": {"type": "int", "low": 5, "high": 100, "step": 5},
        },
    },
    "lasso": {
        "training": {
            "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        },
        "model_params": {
            "alpha": {"type": "float", "low": 1e-4, "high": 1.0, "log": True},
        },
    },
}


def get_default_search_space(model_type: str) -> Dict[str, Dict[str, Any]]:
    """Return the default search space for a given model type.

    Args:
        model_type: Model architecture identifier (e.g. "mlp", "transformer", "lambdamart").

    Returns:
        Dictionary with "training" and "model_params" sections.

    Raises:
        ValueError: If no default search space exists for the model type.
    """
    key = model_type.lower()
    if key not in _DEFAULT_SEARCH_SPACES:
        raise ValueError(
            f"No default search space for model type '{model_type}'. "
            f"Available: {list(_DEFAULT_SEARCH_SPACES)}. "
            f"Please provide a custom search_space."
        )
    return copy.deepcopy(_DEFAULT_SEARCH_SPACES[key])


def _sample_random_param(spec: Dict[str, Any], rng: random.Random) -> Any:
    """Sample a single hyperparameter value from its specification using random sampling.

    Args:
        spec: Parameter specification dict with "type" and range/choice keys.
        rng: Random number generator instance.

    Returns:
        Sampled parameter value.
    """
    ptype = spec["type"]
    if ptype == "float":
        low, high = spec["low"], spec["high"]
        if spec.get("log", False):
            return float(np.exp(rng.uniform(np.log(low), np.log(high))))
        return rng.uniform(low, high)
    elif ptype == "int":
        low, high = spec["low"], spec["high"]
        step = spec.get("step", 1)
        n_steps = (high - low) // step
        return low + rng.randint(0, n_steps) * step
    elif ptype == "categorical":
        choices = spec["choices"]
        return choices[rng.randint(0, len(choices) - 1)]
    else:
        raise ValueError(f"Unknown param type: {ptype}")


def _generate_grid_values(spec: Dict[str, Any]) -> List[Any]:
    """Generate all possible values for a single hyperparameter for grid search.

    Args:
        spec: Parameter specification dict with "type" and range/choice keys.

    Returns:
        List of all values in the grid for this parameter.
    """
    ptype = spec["type"]
    if ptype == "float":
        low, high = spec["low"], spec["high"]
        n_points = spec.get("n_points", 5)
        if spec.get("log", False):
            return np.logspace(np.log10(low), np.log10(high), n_points).tolist()
        return np.linspace(low, high, n_points).tolist()
    elif ptype == "int":
        low, high = spec["low"], spec["high"]
        step = spec.get("step", 1)
        return list(range(low, high + 1, step))
    elif ptype == "categorical":
        return list(spec["choices"])
    else:
        raise ValueError(f"Unknown param type: {ptype}")


def _make_optuna_suggestion(trial, key: str, spec: Dict[str, Any]) -> Any:
    """Sample a hyperparameter using Optuna's trial suggest API.

    Handles type conversion for list choices (tuples → lists) automatically.

    Args:
        trial: Optuna Trial object.
        key: Parameter name.
        spec: Parameter specification dict.

    Returns:
        Sampled parameter value.
    """
    import optuna

    ptype = spec["type"]
    if ptype == "float":
        return trial.suggest_float(key, spec["low"], spec["high"], log=spec.get("log", False))
    elif ptype == "int":
        return trial.suggest_int(key, spec["low"], spec["high"], step=spec.get("step", 1))
    elif ptype == "categorical":
        choices = spec["choices"]
        needs_list_conversion = any(isinstance(c, list) for c in choices)
        if needs_list_conversion:
            tuple_choices = [tuple(c) if isinstance(c, list) else c for c in choices]
            chosen = trial.suggest_categorical(key, tuple_choices)
            return list(chosen) if isinstance(chosen, tuple) else chosen
        return trial.suggest_categorical(key, choices)
    else:
        raise ValueError(f"Unknown param type: {ptype}")


def _extract_best_metric(history: dict, metric: str, direction: str) -> float:
    """Extract the best metric value from a training history dict.

    Args:
        history: History dict returned by model.fit().
        metric: Metric key to extract (e.g. "val_top1_above_mean_rate").
        direction: "maximize" or "minimize".

    Returns:
        Best metric value across all epochs.
    """
    if metric not in history or not history[metric]:
        return float("-inf") if direction == "maximize" else float("inf")

    values = history[metric]
    if direction == "maximize":
        return float(max(values))
    return float(min(values))


def _build_config_from_params(
        base_config: Config,
        training_params: Dict[str, Any],
        model_params: Dict[str, Any],
        disable_torch_compile: bool,
) -> Config:
    """Create a new Config by overriding base_config with sampled hyperparameters.

    Args:
        base_config: Base configuration to modify.
        training_params: Sampled training hyperparameters.
        model_params: Sampled model hyperparameters.
        disable_torch_compile: Whether to force torch_compile=False.

    Returns:
        Deep-copied Config with overridden parameters.
    """
    cfg = copy.deepcopy(base_config)

    for key, value in training_params.items():
        if hasattr(cfg.training, key):
            setattr(cfg.training, key, value)

    for key, value in model_params.items():
        cfg.models[0].params[key] = value

    if disable_torch_compile:
        cfg.training.torch_compile = False

    return cfg


class HyperparamTuner:
    """Automated hyperparameter tuner for model_trainer models.

    Supports random search, grid search, and Optuna-based Bayesian optimization.
    Optimizes the early-stopping metric (auto-detected per model type by default)
    to find the best hyperparameter configuration.

    Data is loaded once and reused across all trials for efficiency.

    Args:
        base_config: Base Config object providing data, model type, and default hyperparameters.
        search_space: Custom search space dict with "training" and "model_params" sections.
                      If None, uses the default search space for the model type.
        method: Search strategy — "random" | "grid" | "optuna".
        n_trials: Number of trials for random/optuna search. Ignored for grid search.
        metric: History key to optimize. "auto" auto-detects based on model type
                (val_top1_above_mean_rate for mlp/transformer/lambdamart, val_ic for gru/dlinear/lgbm).
        direction: Optimization direction — "maximize" or "minimize".
        seed: Random seed for reproducibility.
        save_dir: Directory to save tuning results.
        disable_torch_compile: Force torch_compile=False during tuning to reduce overhead.
        optuna_startup_trials: Number of initial random trials before TPE kicks in (Optuna only).
                              Set to 0 to use TPE from the very first trial.
        optuna_gamma: Ratio of trials used to model the objective function for TPE (Optuna only).
                     Values in (0, 1]. Default is 0.25. Set to None to use Optuna default.
    """

    def __init__(
            self,
            base_config: Config,
            search_space: Optional[Dict[str, Dict[str, Any]]] = None,
            method: str = "random",
            n_trials: int = 50,
            metric: str = "auto",
            direction: str = "maximize",
            seed: int = 42,
            save_dir: str = "tuner_output",
            disable_torch_compile: bool = True,
            optuna_startup_trials: int = 10,
            optuna_gamma: Optional[float] = None,
    ) -> None:
        if direction not in ("maximize", "minimize"):
            raise ValueError(f"direction must be 'maximize' or 'minimize', got '{direction}'")
        if method not in ("random", "grid", "optuna"):
            raise ValueError(f"method must be 'random', 'grid', or 'optuna', got '{method}'")

        self.base_config = base_config
        self.method = method
        self.n_trials = n_trials
        self.direction = direction
        self.seed = seed
        self.save_dir = save_dir
        self.disable_torch_compile = disable_torch_compile
        self.optuna_startup_trials = optuna_startup_trials
        self.optuna_gamma = optuna_gamma

        model_type = base_config.models[0].type.lower()

        if metric == "auto":
            self.metric = _DEFAULT_METRICS.get(model_type, "val_top1_above_mean_rate")
        else:
            self.metric = metric

        if search_space is None:
            self.search_space = get_default_search_space(model_type)
        else:
            self.search_space = search_space

        self.trial_results: List[Dict[str, Any]] = []

    def run(self) -> Dict[str, Any]:
        """Execute the full tuning pipeline: load data, search, save results.

        Returns:
            Result dict with keys:
                - best_metric: Best metric value found.
                - best_trial: Trial number that achieved the best metric.
                - best_config: Config object with the best hyperparameters.
                - best_params: Dict of the best training and model parameters.
                - n_trials: Total number of trials completed.
                - results_df: DataFrame with all trial results.
        """
        print(f"\n{'=' * 60}")
        print(f"Hyperparameter Tuning")
        print(f"{'=' * 60}")
        print(f"  Method       : {self.method}")
        print(f"  N trials     : {self.n_trials}")
        print(f"  Metric       : {self.metric} ({self.direction})")
        print(f"  Model type   : {self.base_config.models[0].type}")
        print(f"  Seed         : {self.seed}")
        print(f"  Save dir     : {self.save_dir}")
        print(f"  Disable compile: {self.disable_torch_compile}")
        if self.method == "optuna":
            print(f"  Startup trials: {self.optuna_startup_trials}")
        print(f"{'=' * 60}\n")

        cfg = self.base_config
        data_cfg = cfg.data

        data_load_start = time.time()
        print("Loading data...")
        df_train, df_val, _, factor_list, norm_stats = load_all_splits(
            data_root=data_cfg.data_root,
            train_dates=data_cfg.train_dates,
            val_dates=data_cfg.val_dates,
            test_dates="",
            factor_list=data_cfg.factor_list,
            y_list=data_cfg.y_list,
            abtypes=data_cfg.abtypes,
            codes=data_cfg.codes,
            normalize=data_cfg.normalize,
            winsorize_std=data_cfg.winsorize_std,
        )
        print(f"[Data] Loaded in {time.time() - data_load_start:.1f}s")
        print(f"[Data] train: {len(df_train):,} rows, val: {len(df_val):,} rows\n")

        if df_train.empty:
            raise RuntimeError("Training data is empty. Check data_root and date ranges.")

        y_col = data_cfg.y_list[0]

        search_start = time.time()
        if self.method == "random":
            result = self._run_random(df_train, df_val, factor_list, y_col)
        elif self.method == "grid":
            result = self._run_grid(df_train, df_val, factor_list, y_col)
        elif self.method == "optuna":
            result = self._run_optuna(df_train, df_val, factor_list, y_col)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        total_duration = time.time() - search_start
        result["total_duration"] = total_duration
        result["results_df"] = pd.DataFrame(self.trial_results)
        result["df_train"] = df_train
        result["df_val"] = df_val
        result["factor_list"] = factor_list
        result["norm_stats"] = norm_stats

        self._save_results(result)

        print(f"\n{'=' * 60}")
        print(f"Tuning Complete")
        print(f"{'=' * 60}")
        print(f"  Best {self.metric}: {result['best_metric']:.4f}")
        print(f"  Best trial   : {result['best_trial']}")
        print(f"  Total time   : {total_duration:.1f}s")
        print(f"  Best params  :")
        for section, params in result.get("best_params", {}).items():
            print(f"    {section}:")
            for k, v in params.items():
                print(f"      {k}: {v}")
        print(f"{'=' * 60}\n")

        return result

    def _train_and_evaluate(
            self,
            cfg: Config,
            df_train: pd.DataFrame,
            df_val: pd.DataFrame,
            factor_list: List[str],
            y_col: str,
    ) -> Tuple[float, dict, Any]:
        """Train a model with the given config and return (best_metric, history, model).

        Args:
            cfg: Config with sampled hyperparameters.
            df_train: Training DataFrame.
            df_val: Validation DataFrame.
            factor_list: Feature column names.
            y_col: Label column name.

        Returns:
            Tuple of (best_metric_value, history_dict, model_object).
        """
        model_entry = cfg.models[0]
        model = build_model(
            model_type=model_entry.type, name=model_entry.name, **model_entry.params
        )

        history = model.fit(
            df_train=df_train,
            factor_list=factor_list,
            y_col=y_col,
            df_val=df_val if not df_val.empty else None,
            training_cfg=cfg.training,
        )

        best_metric = _extract_best_metric(history, self.metric, self.direction)

        return best_metric, history, model

    def _collect_params_from_config(self, cfg: Config) -> Dict[str, Dict[str, Any]]:
        """Extract tuned parameter values from a Config object.

        Args:
            cfg: Config to extract parameters from.

        Returns:
            Dict with "training" and "model_params" sections.
        """
        training_params = {}
        for key in self.search_space.get("training", {}):
            if hasattr(cfg.training, key):
                training_params[key] = getattr(cfg.training, key)

        model_params = {}
        for key in self.search_space.get("model_params", {}):
            model_params[key] = cfg.models[0].params.get(key)

        return {"training": training_params, "model_params": model_params}

    def _run_random(
            self,
            df_train: pd.DataFrame,
            df_val: pd.DataFrame,
            factor_list: List[str],
            y_col: str,
    ) -> Dict[str, Any]:
        """Run random search over the hyperparameter space.

        Args:
            df_train: Training DataFrame.
            df_val: Validation DataFrame.
            factor_list: Feature column names.
            y_col: Label column name.

        Returns:
            Result dict with best metric, trial number, and config.
        """
        rng = random.Random(self.seed)
        worst = float("-inf") if self.direction == "maximize" else float("inf")

        best_metric = worst
        best_config: Optional[Config] = None
        best_model = None
        best_trial = -1

        training_specs = self.search_space.get("training", {})
        model_specs = self.search_space.get("model_params", {})

        for trial_idx in range(self.n_trials):
            trial_start = time.time()

            training_params = {k: _sample_random_param(v, rng) for k, v in training_specs.items()}
            model_params = {k: _sample_random_param(v, rng) for k, v in model_specs.items()}

            cfg = _build_config_from_params(
                self.base_config, training_params, model_params, self.disable_torch_compile
            )

            try:
                metric_value, history, model = self._train_and_evaluate(
                    cfg, df_train, df_val, factor_list, y_col
                )
                is_best = (
                        (self.direction == "maximize" and metric_value >= best_metric)
                        or (self.direction == "minimize" and metric_value <= best_metric)
                )

                if is_best:
                    best_metric = metric_value
                    best_config = cfg
                    best_model = model
                    best_trial = trial_idx + 1
                else:
                    del model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                duration = time.time() - trial_start
                self.trial_results.append({
                    "trial": trial_idx + 1,
                    self.metric: metric_value,
                    "is_best": is_best,
                    "duration_s": duration,
                    **{f"training.{k}": v for k, v in training_params.items()},
                    **{f"model_params.{k}": v for k, v in model_params.items()},
                })

                print(
                    f"  Trial {trial_idx + 1}/{self.n_trials}: "
                    f"{self.metric}={metric_value:.4f} "
                    f"{'★ BEST' if is_best else ''} | "
                    f"time={duration:.1f}s"
                )

            except Exception as exc:
                duration = time.time() - trial_start
                self.trial_results.append({
                    "trial": trial_idx + 1,
                    self.metric: None,
                    "is_best": False,
                    "duration_s": duration,
                    "error": str(exc),
                    **{f"training.{k}": v for k, v in training_params.items()},
                    **{f"model_params.{k}": v for k, v in model_params.items()},
                })
                print(f"  Trial {trial_idx + 1}/{self.n_trials}: FAILED — {exc}")

        best_params = self._collect_params_from_config(best_config) if best_config else {}
        return {
            "best_metric": best_metric,
            "best_trial": best_trial,
            "best_config": best_config,
            "best_model": best_model,
            "best_params": best_params,
            "n_trials": self.n_trials,
        }

    def _run_grid(
            self,
            df_train: pd.DataFrame,
            df_val: pd.DataFrame,
            factor_list: List[str],
            y_col: str,
    ) -> Dict[str, Any]:
        """Run grid search over all hyperparameter combinations.

        Args:
            df_train: Training DataFrame.
            df_val: Validation DataFrame.
            factor_list: Feature column names.
            y_col: Label column name.

        Returns:
            Result dict with best metric, trial number, and config.
        """
        all_grids: Dict[str, List[Any]] = {}
        for key, spec in self.search_space.get("training", {}).items():
            all_grids[f"training.{key}"] = _generate_grid_values(spec)
        for key, spec in self.search_space.get("model_params", {}).items():
            all_grids[f"model_params.{key}"] = _generate_grid_values(spec)

        keys = list(all_grids.keys())
        value_lists = list(all_grids.values())
        total = 1
        for vl in value_lists:
            total *= len(vl)

        print(f"  Grid search: {total} combinations")

        worst = float("-inf") if self.direction == "maximize" else float("inf")
        best_metric = worst
        best_config: Optional[Config] = None
        best_model = None
        best_trial = -1

        for trial_idx, combo in enumerate(itertools.product(*value_lists)):
            trial_start = time.time()

            training_params = {}
            model_params = {}
            for key, value in zip(keys, combo):
                section, param = key.split(".", 1)
                if section == "training":
                    training_params[param] = value
                elif section == "model_params":
                    model_params[param] = value

            cfg = _build_config_from_params(
                self.base_config, training_params, model_params, self.disable_torch_compile
            )

            try:
                metric_value, history, model = self._train_and_evaluate(
                    cfg, df_train, df_val, factor_list, y_col
                )
                is_best = (
                        (self.direction == "maximize" and metric_value >= best_metric)
                        or (self.direction == "minimize" and metric_value <= best_metric)
                )

                if is_best:
                    best_metric = metric_value
                    best_config = cfg
                    best_model = model
                    best_trial = trial_idx + 1
                else:
                    del model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                duration = time.time() - trial_start
                self.trial_results.append({
                    "trial": trial_idx + 1,
                    self.metric: metric_value,
                    "is_best": is_best,
                    "duration_s": duration,
                    **{f"training.{k}": v for k, v in training_params.items()},
                    **{f"model_params.{k}": v for k, v in model_params.items()},
                })

                print(
                    f"  Trial {trial_idx + 1}/{total}: "
                    f"{self.metric}={metric_value:.4f} "
                    f"{'★ BEST' if is_best else ''} | "
                    f"time={duration:.1f}s"
                )

            except Exception as exc:
                duration = time.time() - trial_start
                self.trial_results.append({
                    "trial": trial_idx + 1,
                    self.metric: None,
                    "is_best": False,
                    "duration_s": duration,
                    "error": str(exc),
                    **{f"training.{k}": v for k, v in training_params.items()},
                    **{f"model_params.{k}": v for k, v in model_params.items()},
                })
                print(f"  Trial {trial_idx + 1}/{total}: FAILED — {exc}")

        best_params = self._collect_params_from_config(best_config) if best_config else {}
        return {
            "best_metric": best_metric,
            "best_trial": best_trial,
            "best_config": best_config,
            "best_model": best_model,
            "best_params": best_params,
            "n_trials": total,
        }

    def _run_optuna(
            self,
            df_train: pd.DataFrame,
            df_val: pd.DataFrame,
            factor_list: List[str],
            y_col: str,
    ) -> Dict[str, Any]:
        """Run Optuna Bayesian optimization (TPE sampler).

        Args:
            df_train: Training DataFrame.
            df_val: Validation DataFrame.
            factor_list: Feature column names.
            y_col: Label column name.

        Returns:
            Result dict with best metric, trial number, and config.

        Raises:
            ImportError: If optuna is not installed.
        """
        try:
            import optuna
        except ImportError:
            raise ImportError(
                "optuna is required for Bayesian optimization. "
                "Install with: pip install optuna"
            )

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        gamma_param = (lambda n: max(int(n * self.optuna_gamma), 1)) if self.optuna_gamma is not None else None
        sampler_kwargs = {"seed": self.seed, "n_startup_trials": self.optuna_startup_trials}
        if gamma_param is not None:
            sampler_kwargs["gamma"] = gamma_param
        sampler = optuna.samplers.TPESampler(**sampler_kwargs)
        study = optuna.create_study(direction=self.direction, sampler=sampler)

        training_specs = self.search_space.get("training", {})
        model_specs = self.search_space.get("model_params", {})

        worst = float("-inf") if self.direction == "maximize" else float("inf")
        best_so_far = [worst]
        best_model_holder = [None]

        def objective(trial: optuna.Trial) -> float:
            training_params = {
                k: _make_optuna_suggestion(trial, f"training.{k}", v)
                for k, v in training_specs.items()
            }
            model_params = {
                k: _make_optuna_suggestion(trial, f"model_params.{k}", v)
                for k, v in model_specs.items()
            }

            cfg = _build_config_from_params(
                self.base_config, training_params, model_params, self.disable_torch_compile
            )

            trial_start = time.time()
            try:
                metric_value, history, model = self._train_and_evaluate(
                    cfg, df_train, df_val, factor_list, y_col
                )
                is_best = (
                        (self.direction == "maximize" and metric_value >= best_so_far[0])
                        or (self.direction == "minimize" and metric_value <= best_so_far[0])
                )
                if is_best:
                    best_so_far[0] = metric_value
                    best_model_holder[0] = model
                else:
                    del model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                duration = time.time() - trial_start
                self.trial_results.append({
                    "trial": trial.number + 1,
                    self.metric: metric_value,
                    "is_best": is_best,
                    "duration_s": duration,
                    **{f"training.{k}": v for k, v in training_params.items()},
                    **{f"model_params.{k}": v for k, v in model_params.items()},
                })

                print(
                    f"  Trial {trial.number + 1}: "
                    f"{self.metric}={metric_value:.4f} "
                    f"{'★ BEST' if is_best else ''} | "
                    f"time={duration:.1f}s"
                )

                return metric_value

            except Exception as exc:
                duration = time.time() - trial_start
                self.trial_results.append({
                    "trial": trial.number + 1,
                    self.metric: None,
                    "is_best": False,
                    "duration_s": duration,
                    "error": str(exc),
                    **{f"training.{k}": v for k, v in training_params.items()},
                    **{f"model_params.{k}": v for k, v in model_params.items()},
                })
                print(f"  Trial {trial.number + 1}: FAILED — {exc}")
                return worst

        study.optimize(objective, n_trials=self.n_trials)

        best_trial_result = study.best_trial
        best_cfg = copy.deepcopy(self.base_config)

        for key, spec in training_specs.items():
            param_key = f"training.{key}"
            if param_key in best_trial_result.params:
                value = best_trial_result.params[param_key]
                if isinstance(value, tuple):
                    value = list(value)
                setattr(best_cfg.training, key, value)

        for key, spec in model_specs.items():
            param_key = f"model_params.{key}"
            if param_key in best_trial_result.params:
                value = best_trial_result.params[param_key]
                if isinstance(value, tuple):
                    value = list(value)
                best_cfg.models[0].params[key] = value

        if self.disable_torch_compile:
            best_cfg.training.torch_compile = False

        best_params = self._collect_params_from_config(best_cfg)

        return {
            "best_metric": best_trial_result.value,
            "best_trial": best_trial_result.number + 1,
            "best_config": best_cfg,
            "best_model": best_model_holder[0],
            "best_params": best_params,
            "n_trials": len(study.trials),
        }

    def _save_results(self, result: Dict[str, Any]) -> None:
        """Save tuning results to disk.

        Saves:
            - tuning_results.csv: All trial results.
            - best_config.json: Best hyperparameters and metric.
            - best_config.yaml: Best config in YAML format (if pyyaml available).
            - best model weights file (if best_model is available).

        Args:
            result: Result dict from the search method.
        """
        os.makedirs(self.save_dir, exist_ok=True)

        results_df = result.get("results_df")
        if results_df is not None and not results_df.empty:
            results_df.to_csv(os.path.join(self.save_dir, "tuning_results.csv"), index=False)

        best_config_dict = {
            "best_metric": result.get("best_metric"),
            "best_trial": result.get("best_trial"),
            "n_trials": result.get("n_trials"),
            "total_duration_s": result.get("total_duration"),
            "metric": self.metric,
            "direction": self.direction,
            "method": self.method,
            "params": result.get("best_params", {}),
        }

        with open(os.path.join(self.save_dir, "best_config.json"), "w") as fp:
            json.dump(best_config_dict, fp, indent=2, default=str)

        try:
            import yaml
            best_cfg = result.get("best_config")
            if best_cfg is not None:
                yaml_dict = {
                    "data": {
                        "data_root": best_cfg.data.data_root,
                        "train_dates": best_cfg.data.train_dates,
                        "val_dates": best_cfg.data.val_dates,
                        "test_dates": best_cfg.data.test_dates,
                        "y_list": best_cfg.data.y_list,
                        "factor_list": best_cfg.data.factor_list,
                        "normalize": best_cfg.data.normalize,
                        "winsorize_std": best_cfg.data.winsorize_std,
                    },
                    "models": [
                        {
                            "type": m.type,
                            "name": m.name,
                            "params": m.params,
                        }
                        for m in best_cfg.models
                    ],
                    "training": {
                        k: getattr(best_cfg.training, k)
                        for k in [
                            "epochs", "batch_size", "learning_rate",
                            "early_stopping_patience", "min_epochs", "min_delta",
                            "device", "loss_type", "topk", "rank_before_softmax",
                            "use_hard_label", "label_smoothing", "torch_compile",
                        ]
                    },
                    "output": {
                        "save_dir": best_cfg.output.save_dir,
                        "experiment_name": best_cfg.output.experiment_name,
                        "upload_to_hdfs": best_cfg.output.upload_to_hdfs,
                    },
                }
                with open(os.path.join(self.save_dir, "best_config.yaml"), "w") as fp:
                    yaml.dump(yaml_dict, fp, default_flow_style=False, allow_unicode=True)
        except ImportError:
            pass

        best_model = result.get("best_model")
        if best_model is not None:
            model_type = self.base_config.models[0].type.lower()
            model_file_ext = {
                "lgbm": ".txt", "mlp": ".pth", "transformer": ".pth",
                "gru": ".pth", "dlinear": ".pth", "lambdamart": ".txt",
            }.get(model_type, ".bin")
            model_path = os.path.join(self.save_dir, f"best_model{model_file_ext}")
            best_model.save(path=model_path)
            print(f"  Best model saved to: {model_path}")

        print(f"  Results saved to: {self.save_dir}")
