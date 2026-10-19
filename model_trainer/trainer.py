"""Main training orchestrator.

Flow:
    1. Load & pre-process all three data splits once.
    2. For each label in y_list:
       For each model in config.models:
           a. Build model.
           b. Fit on train (with val for early-stopping).
           c. Predict on train / val / test.
           d. Compute metrics.
           e. Save plots: training curve, feature importance.
           f. Optionally save model weights.
    3. Write a summary CSV.
"""
from __future__ import annotations

import gc
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow.fs as fs
import torch

from .config import Config
from .data import load_all_splits, load_split
from .metrics import evaluate, build_prior_topk_abtypes, TOPK_VALUES
from .models import build_model
from .plots import (
    plot_feature_importance,
    plot_training_curve,
    plot_abtype_distribution,
)


class Trainer:
    """High-level trainer that wires data → models → evaluation → plots.

    Orchestrates the full experiment pipeline: data loading, model training,
    prediction, metric evaluation, plot generation, and result persistence.

    Args:
        config: Root Config object containing all experiment settings.
    """

    def __init__(self, config: Config) -> None:
        self.cfg = config

    def run(self) -> pd.DataFrame:
        """Execute the full experiment pipeline.

        Returns:
            Summary DataFrame with one row per model × y_col combination.
        """
        print(f'Current working directory: {os.getcwd()}')
        torch.set_float32_matmul_precision('high')

        cfg = self.cfg
        data_cfg = cfg.data

        # ── 1. Load data ─────────────────────────────────────────────────
        data_load_start = time.time()
        print("\n" + "=" * 60)
        print(f"Experiment : {cfg.output.experiment_name}")
        print(f"Data root  : {data_cfg.data_root}")
        print(f"Normalize  : {data_cfg.normalize}")
        print("=" * 60)

        df_train, df_val, _, factor_list, norm_stats = load_all_splits(
            data_root=data_cfg.data_root,
            train_dates=data_cfg.train_dates,
            val_dates=data_cfg.val_dates,
            test_dates='',
            factor_list=data_cfg.factor_list,
            y_list=data_cfg.y_list,
            abtypes=data_cfg.abtypes,
            codes=data_cfg.codes,
            normalize=data_cfg.normalize,
            winsorize_std=data_cfg.winsorize_std,
        )
        print(f"[Data] Loaded in {time.time() - data_load_start:.1f}s\n")

        if df_train.empty:
            raise RuntimeError("Training data is empty. Check data_root and date ranges.")

        # ── 2. Per-label, per-model loop ──────────────────────────────────
        summary_rows: List[dict] = []

        assert len(data_cfg.y_list) == 1

        for y_col in data_cfg.y_list:
            print("\n" + "─" * 60)
            print(f"  Label: {y_col}")
            print("─" * 60)

            model_test_metrics: Dict[str, dict] = {}

            assert len(cfg.models) == 1

            for model_entry in cfg.models:
                print(f"\n  ► Model: {model_entry.name} ({model_entry.type})")
                output_dir = os.path.join(
                    cfg.output.save_dir, cfg.output.experiment_name, y_col, model_entry.name
                )
                os.makedirs(output_dir, exist_ok=True)

                # ── a. Build ──────────────────────────────────────────────
                model = build_model(
                    model_type=model_entry.type, name=model_entry.name, **model_entry.params
                )

                # ── b. Fit ────────────────────────────────────────────────
                fit_start = time.time()
                history = model.fit(
                    df_train=df_train,
                    factor_list=factor_list,
                    y_col=y_col,
                    df_val=df_val if not df_val.empty else None,
                    training_cfg=cfg.training,
                )
                print(f"  Training done in {time.time() - fit_start:.1f}s")

                # ── c. Save model ─────────────────────────────────────────
                if cfg.output.save_model:
                    model_file_ext = {
                        "lgbm": ".txt", "mlp": ".pth", "transformer": ".pth",
                        "gru": ".pth", "dlinear": ".pth", "lambdamart": ".txt",
                    }.get(model_entry.type, ".bin")
                    model.save(path=os.path.join(output_dir, f"model{model_file_ext}"))

                # ── d. Predict ────────────────────────────────────────────
                def _safe_predict(df: pd.DataFrame) -> Optional[np.ndarray]:
                    """Attempt prediction, returning None on failure."""
                    if df.empty:
                        return None
                    try:
                        return model.predict(df=df, factor_list=factor_list)
                    except Exception as exc:
                        print(f"  [Warning] Prediction failed: {exc}")
                        return None

                # ── d. Metrics ────────────────────────────────────────────
                prior_topk_abtypes = None
                df_union = (
                    pd.concat([df_train, df_val], ignore_index=True)
                    if not df_val.empty
                    else df_train
                )
                if "abtype" in df_union.columns:
                    prior_topk_abtypes = build_prior_topk_abtypes(df=df_union, y_col=y_col)
                del df_union

                pred_train = _safe_predict(df=df_train)
                pred_val = _safe_predict(df=df_val)
                print("  Train metrics:")
                metrics_train = (
                    evaluate(
                        y_pred=pred_train, y_true=df_train[y_col].values,
                        split_name="train", df=df_train, topk=cfg.training.topk,
                        prior_topk_abtypes=prior_topk_abtypes,
                    )
                    if pred_train is not None
                    else {}
                )
                print("  Val metrics:")
                metrics_val = (
                    evaluate(
                        y_pred=pred_val, y_true=df_val[y_col].values,
                        split_name="val", df=df_val, topk=cfg.training.topk,
                        prior_topk_abtypes=prior_topk_abtypes,
                    )
                    if pred_val is not None and not df_val.empty
                    else {}
                )

                if pred_train is not None and not df_train.empty:
                    plot_abtype_distribution(
                        df=df_train,
                        predictions=pred_train,
                        y_col=y_col,
                        title=f"{model_entry.name} | {y_col} — Abtype Distribution (Train)",
                        save_path=os.path.join(output_dir, "abtype_distribution_train.png"),
                    )

                if pred_val is not None and not df_val.empty:
                    plot_abtype_distribution(
                        df=df_val,
                        predictions=pred_val,
                        y_col=y_col,
                        title=f"{model_entry.name} | {y_col} — Abtype Distribution (Val)",
                        save_path=os.path.join(output_dir, "abtype_distribution_val.png"),
                    )

                del df_train, df_val
                gc.collect()

                df_test, _ = load_split(
                    data_root=data_cfg.data_root,
                    date_range=data_cfg.test_dates,
                    factor_list=factor_list,
                    y_list=data_cfg.y_list,
                    abtypes=data_cfg.abtypes,
                    codes=data_cfg.codes,
                    normalize=data_cfg.normalize,
                    winsorize_std=data_cfg.winsorize_std,
                    prev_stats=norm_stats,
                )
                print(f"[Data] Loaded → test:{len(df_test):,} rows")

                pred_test = _safe_predict(df=df_test)

                print("  Test metrics:")
                metrics_test = (
                    evaluate(
                        y_pred=pred_test, y_true=df_test[y_col].values,
                        split_name="test", df=df_test, topk=cfg.training.topk,
                        prior_topk_abtypes=prior_topk_abtypes,
                    )
                    if pred_test is not None and not df_test.empty
                    else {}
                )

                model_test_metrics[model_entry.name] = metrics_test

                # ── e. Plots ──────────────────────────────────────────────
                if history:
                    plot_training_curve(
                        history=history,
                        title=f"{model_entry.name} | {y_col} — Training",
                        save_path=os.path.join(output_dir, "training_curve.png"),
                    )

                feature_importance = model.feature_importance()
                if feature_importance is not None:
                    plot_feature_importance(
                        importance=feature_importance,
                        title=f"{model_entry.name} | {y_col} — Feature Importance",
                        save_path=os.path.join(output_dir, "feature_importance.png"),
                    )

                if not df_test.empty and pred_test is not None:
                    plot_abtype_distribution(
                        df=df_test,
                        predictions=pred_test,
                        y_col=y_col,
                        title=f"{model_entry.name} | {y_col} — Abtype Distribution (Test)",
                        save_path=os.path.join(output_dir, "abtype_distribution_test.png"),
                    )

                # ── g. Save test set predictions ─────────────────────────
                if not df_test.empty and pred_test is not None:
                    print("  Saving test set predictions (Optimized Mode)...")

                    prediction_df = pd.DataFrame(
                        {
                            'date': df_test['date'].values,
                            'code': df_test['code'].values,
                            'abtype': df_test['abtype'].values,
                            f'{y_col}_pred': pred_test,
                            'time_sec': df_test['time_sec'].values,
                        },
                        index=df_test.index,
                    )

                    del df_test, pred_test
                    gc.collect()

                    if cfg.output.upload_to_hdfs:

                        def _process_and_upload(
                                date_str: str,
                                code: int,
                                abtype: int,
                                df_slice: pd.DataFrame,
                        ) -> None:
                            """Upload a single prediction slice to HDFS.

                            Each slice is saved as a local parquet file then
                            copied to HDFS via pyarrow.

                            Args:
                                date_str: Date string for the HDFS path.
                                code: Instrument code for the filename.
                                abtype: Abtype code for the filename.
                                df_slice: DataFrame slice to upload.
                            """
                            try:
                                os.environ["HADOOP_USER_NAME"] = cfg.output.hdfs_user
                                hdfs_fs = fs.HadoopFileSystem.from_uri(cfg.output.hdfs_uri)
                                local_fs = fs.LocalFileSystem()

                                remote_dir = os.path.join(
                                    cfg.output.hdfs_remote_root,
                                    cfg.output.experiment_name,
                                    y_col,
                                    model_entry.name,
                                    "predictions",
                                    date_str,
                                )
                                file_name = f"{int(code)}_{int(abtype)}.parquet"
                                remote_path = os.path.join(remote_dir, file_name)

                                local_temp = os.path.abspath(
                                    f"/tmp/pred_{date_str}_{int(code)}_{int(abtype)}.parquet"
                                )

                                df_slice.to_parquet(local_temp, index=False)

                                hdfs_fs.create_dir(remote_dir)
                                fs.copy_files(
                                    source=local_temp,
                                    destination=remote_path,
                                    source_filesystem=local_fs,
                                    destination_filesystem=hdfs_fs,
                                )

                                if os.path.exists(local_temp):
                                    os.remove(local_temp)

                            except Exception as exc:
                                print(f"  [Thread Error] Failed {date_str}/{code}/{abtype}: {exc}")

                        upload_tasks = []
                        for (date_val, code_val, abtype_val), group_df in prediction_df.groupby(
                            ['date', 'code', 'abtype']
                        ):
                            date_str = str(date_val)
                            if not group_df.empty:
                                upload_tasks.append((date_str, code_val, abtype_val, group_df))

                        if upload_tasks:
                            max_workers = min(os.cpu_count() or 4, len(upload_tasks), 64)
                            print(
                                f"  [Upload] Starting thread pool with {max_workers} workers "
                                f"for {len(upload_tasks)} files..."
                            )

                            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                                futures = [
                                    executor.submit(_process_and_upload, *task)
                                    for task in upload_tasks
                                ]
                                for future in as_completed(futures):
                                    pass

                            print(
                                f"  [Upload] Predictions saved to HDFS: "
                                f"{os.path.join(cfg.output.hdfs_remote_root, cfg.output.experiment_name, y_col, model_entry.name, 'predictions')}"
                            )
                        else:
                            print("  [Upload] No valid prediction files generated.")

                # Save metrics JSON
                def _json_safe(value):
                    """Convert numpy/pandas types to JSON-serializable Python types."""
                    if pd.isna(value):
                        return None
                    if isinstance(value, (np.floating, np.integer)):
                        return float(value)
                    return value

                metrics_output = {
                    "train": {
                        k: _json_safe(v) for k, v in metrics_train.items()
                        if not isinstance(v, pd.Series)
                    },
                    "val": {
                        k: _json_safe(v) for k, v in metrics_val.items()
                        if not isinstance(v, pd.Series)
                    },
                    "test": {
                        k: _json_safe(v) for k, v in metrics_test.items()
                        if not isinstance(v, pd.Series)
                    },
                }

                with open(os.path.join(output_dir, "metrics.json"), "w") as fp:
                    json.dump(metrics_output, fp, indent=2)

                summary_rows.append({
                    "y_col": y_col,
                    "model": model_entry.name,
                    "m_train": metrics_train,
                    "m_val": metrics_val,
                    "m_test": metrics_test,
                })

        # ── 3. Split metrics into category-specific CSVs ──────────────────
        summary_dir = os.path.join(cfg.output.save_dir, cfg.output.experiment_name)
        os.makedirs(summary_dir, exist_ok=True)

        hit_rate_rows = []
        above_mean_rows = []
        vs_prior_rows = []

        for row in summary_rows:
            row_y_col = row["y_col"]
            row_model = row["model"]
            for split_name, metrics_key in [("train", "m_train"), ("val", "m_val"), ("test", "m_test")]:
                split_metrics = row[metrics_key]
                hit_rate_rows.append({
                    "y_col": row_y_col,
                    "model": row_model,
                    "split": split_name,
                    "topk_hit_rate": split_metrics.get("topk_hit_rate", np.nan),
                    "excess_hit_rate": split_metrics.get("excess_hit_rate", np.nan),
                    "avg_pred_rank": split_metrics.get("avg_pred_rank", np.nan),
                })
                above_mean_row = {
                    "y_col": row_y_col, "model": row_model, "split": split_name,
                }
                for topk_val in TOPK_VALUES:
                    above_mean_row[f"top{topk_val}_above_mean_rate"] = split_metrics.get(
                        f"top{topk_val}_above_mean_rate", np.nan
                    )
                above_mean_rows.append(above_mean_row)

                vs_prior_row = {
                    "y_col": row_y_col, "model": row_model, "split": split_name,
                }
                for topk_val in TOPK_VALUES:
                    vs_prior_row[f"top{topk_val}_vs_prior_rate"] = split_metrics.get(
                        f"top{topk_val}_vs_prior_rate", np.nan
                    )
                vs_prior_rows.append(vs_prior_row)

        csv_files = []

        if hit_rate_rows:
            hit_rate_df = pd.DataFrame(hit_rate_rows)
            hit_rate_path = os.path.join(
                summary_dir, f"{cfg.output.experiment_name}_hit_rate_summary.csv"
            )
            hit_rate_df.to_csv(hit_rate_path, index=False)
            csv_files.append(hit_rate_path)

        if above_mean_rows:
            above_mean_df = pd.DataFrame(above_mean_rows)
            above_mean_path = os.path.join(
                summary_dir, f"{cfg.output.experiment_name}_above_mean_summary.csv"
            )
            above_mean_df.to_csv(above_mean_path, index=False)
            csv_files.append(above_mean_path)

        if vs_prior_rows:
            vs_prior_df = pd.DataFrame(vs_prior_rows)
            vs_prior_path = os.path.join(
                summary_dir, f"{cfg.output.experiment_name}_vs_prior_summary.csv"
            )
            vs_prior_df.to_csv(vs_prior_path, index=False)
            csv_files.append(vs_prior_path)

        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        for csv_path in csv_files:
            df_name = os.path.basename(csv_path)
            print(f"\n--- {df_name} ---")
            print(pd.read_csv(csv_path).to_string(index=False))
        print(f"\nResults saved to: {summary_dir}")

        if cfg.output.upload_to_hdfs:
            local_output_dir = os.path.join(cfg.output.save_dir, cfg.output.experiment_name)
            remote_output_dir = os.path.join(cfg.output.hdfs_remote_root, cfg.output.experiment_name)
            print(f"\n[Upload] Uploading to HDFS: {local_output_dir} -> {remote_output_dir}")
            try:
                os.environ["HADOOP_USER_NAME"] = cfg.output.hdfs_user
                hdfs_fs = fs.HadoopFileSystem.from_uri(cfg.output.hdfs_uri)
                fs.copy_files(
                    source=local_output_dir,
                    destination=remote_output_dir,
                    source_filesystem=fs.LocalFileSystem(),
                    destination_filesystem=hdfs_fs,
                )
                print(f"[Upload] Done: {remote_output_dir}")
            except Exception as exc:
                print(f"[Upload] Failed: {exc}")

        return hit_rate_df if hit_rate_rows else pd.DataFrame()
