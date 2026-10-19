"""Unified training runner — reads train_configs.yaml and runs experiments by index.

Supports two modes per experiment:
  - Normal training: Trainer(cfg).run()
  - HPO (Hyperparameter Optimization): HyperparamTuner(cfg, ...).run() + evaluate best on test

Usage:
    # Run a specific experiment by 1-based index
    python train_runner.py --id 1

    # Run multiple experiments sequentially
    python train_runner.py --id 1 2 3

    # SLURM array mode (uses SLURM_ARRAY_TASK_ID environment variable)
    SLURM_ARRAY_TASK_ID=1 python train_runner.py

    # Interactive mode — list experiments and choose
    python train_runner.py

    # Use a different config file
    python train_runner.py --config my_configs.yaml --id 1
"""

import argparse
import os
import sys

import pandas as pd
import yaml

from model_trainer.config import Config
from model_trainer.data import load_split
from model_trainer.metrics import evaluate, build_prior_topk_abtypes
from model_trainer.trainer import Trainer


def split_date_range(date_range: str, train_ratio: float = 0.5, val_ratio: float = 0.25) -> tuple:
    start, end = date_range.split("-")
    dates = pd.date_range(start=start, end=end, freq="D")
    n_dates = len(dates)

    n_train = int(n_dates * train_ratio)
    n_val = int(n_dates * val_ratio)

    train_list = dates[:n_train].strftime("%Y%m%d").tolist()
    val_list = dates[n_train:n_train + n_val].strftime("%Y%m%d").tolist()
    test_list = dates[n_train + n_val:].strftime("%Y%m%d").tolist()

    return f"{train_list[0]}-{train_list[-1]}", f"{val_list[0]}-{val_list[-1]}", f"{test_list[0]}-{test_list[-1]}"


def load_configs(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_config(shared_data: dict, experiment: dict) -> Config:
    date_range = shared_data["date_range"]
    train_ratio = shared_data.get("train_ratio", 0.5)
    val_ratio = shared_data.get("val_ratio", 0.25)
    train_dates, val_dates, test_dates = split_date_range(date_range, train_ratio, val_ratio)

    data_dict = {
        "data_root": shared_data["data_root"],
        "train_dates": train_dates,
        "val_dates": val_dates,
        "test_dates": test_dates,
        "factor_list": shared_data.get("factor_list"),
        "y_list": shared_data.get("y_list"),
        "normalize": shared_data.get("normalize", "none"),
        "winsorize_std": shared_data.get("winsorize_std", 0),
    }
    if "abtypes" in shared_data and shared_data["abtypes"] is not None:
        data_dict["abtypes"] = shared_data["abtypes"]
    if "codes" in shared_data and shared_data["codes"] is not None:
        data_dict["codes"] = shared_data["codes"]

    config_dict = {
        "data": data_dict,
        "models": experiment["models"],
        "training": experiment.get("training", {}),
        "output": experiment.get("output", {}),
    }
    if "filter" in experiment:
        config_dict["filter"] = experiment["filter"]

    return Config.from_dict(config_dict)


def upload_dir_to_hdfs(local_dir: str, cfg: Config, hdfs_subdir: str = None) -> None:
    if not cfg.output.upload_to_hdfs:
        return
    if not os.path.isdir(local_dir):
        print(f"[Upload] Local dir not found, skipping: {local_dir}")
        return
    try:
        from pyarrow import fs as pa_fs
        os.environ["HADOOP_USER_NAME"] = cfg.output.hdfs_user
        hdfs_fs = pa_fs.HadoopFileSystem.from_uri(cfg.output.hdfs_uri)
        if hdfs_subdir:
            remote_root = os.path.join(cfg.output.hdfs_remote_root, hdfs_subdir, os.path.basename(local_dir))
        else:
            remote_root = os.path.join(cfg.output.hdfs_remote_root, os.path.basename(local_dir))
        print(f"[Upload] Uploading to HDFS: {local_dir} -> {remote_root}")
        hdfs_fs.create_dir(remote_root, recursive=True)
        pa_fs.copy_files(
            source=local_dir,
            destination=remote_root,
            source_filesystem=pa_fs.LocalFileSystem(),
            destination_filesystem=hdfs_fs,
        )
        print(f"[Upload] Done: {remote_root}")
    except Exception as exc:
        print(f"[Upload] Failed: {exc}")


def evaluate_best_on_test(best_model, best_cfg, df_train, df_val, factor_list, norm_stats, hpo_save_dir):
    data_cfg = best_cfg.data
    y_col = data_cfg.y_list[0]

    df_union = (
        pd.concat([df_train, df_val], ignore_index=True)
        if not df_val.empty
        else df_train
    )
    prior_topk_abtypes = None
    if "abtype" in df_union.columns:
        prior_topk_abtypes = build_prior_topk_abtypes(df=df_union, y_col=y_col)
    del df_union

    pred_train = best_model.predict(df=df_train, factor_list=factor_list)
    pred_val = best_model.predict(df=df_val, factor_list=factor_list)

    print("  Train metrics (best model):")
    evaluate(
        y_pred=pred_train, y_true=df_train[y_col].values,
        split_name="train", df=df_train, topk=best_cfg.training.topk,
        prior_topk_abtypes=prior_topk_abtypes,
    )
    print("  Val metrics (best model):")
    evaluate(
        y_pred=pred_val, y_true=df_val[y_col].values,
        split_name="val", df=df_val, topk=best_cfg.training.topk,
        prior_topk_abtypes=prior_topk_abtypes,
    )

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
    print(f"[Data] Loaded test: {len(df_test):,} rows")

    pred_test = best_model.predict(df=df_test, factor_list=factor_list)

    print("  Test metrics (best model):")
    test_metrics = evaluate(
        y_pred=pred_test, y_true=df_test[y_col].values,
        split_name="test", df=df_test, topk=best_cfg.training.topk,
        prior_topk_abtypes=prior_topk_abtypes,
    )

    if not df_test.empty and pred_test is not None:
        prediction_df = pd.DataFrame({
            'date': df_test['date'].values,
            'code': df_test['code'].values,
            'abtype': df_test['abtype'].values,
            f'{y_col}_pred': pred_test,
            'time_sec': df_test['time_sec'].values,
        })
        pred_path = os.path.join(hpo_save_dir, "test_predictions.csv")
        prediction_df.to_csv(pred_path, index=False)
        print(f"  Test predictions saved to: {pred_path}")

    return test_metrics


def run_hpo(cfg: Config, shared_hpo: dict) -> None:
    from model_trainer.tuner import HyperparamTuner, get_default_search_space

    method = shared_hpo.get("method", "random")
    n_trials = shared_hpo.get("n_trials", 50)
    metric = shared_hpo.get("metric", "auto")
    direction = shared_hpo.get("direction", "maximize")
    seed = shared_hpo.get("seed", 42)
    save_dir = shared_hpo.get("save_dir", "hpo_output")
    disable_torch_compile = shared_hpo.get("disable_torch_compile", False)
    optuna_startup_trials = shared_hpo.get("optuna_startup_trials", 10)
    optuna_gamma = shared_hpo.get("optuna_gamma", None)

    model_type = cfg.models[0].type
    search_space = get_default_search_space(model_type)

    print(f"\nSearch space for '{model_type}':")
    for section, params in search_space.items():
        print(f"  [{section}]")
        for k, spec in params.items():
            print(f"    {k}: {spec}")
    print()

    exp_save_dir = os.path.join(save_dir, cfg.output.experiment_name)
    tuner = HyperparamTuner(
        base_config=cfg,
        search_space=search_space,
        method=method,
        n_trials=n_trials,
        metric=metric,
        direction=direction,
        seed=seed,
        save_dir=exp_save_dir,
        disable_torch_compile=disable_torch_compile,
        optuna_startup_trials=optuna_startup_trials,
        optuna_gamma=optuna_gamma,
    )

    result = tuner.run()

    best_cfg = result["best_config"]
    best_model = result["best_model"]
    df_train = result["df_train"]
    df_val = result["df_val"]
    factor_list = result["factor_list"]
    norm_stats = result["norm_stats"]

    if best_model is not None and best_cfg is not None:
        print("\n" + "=" * 60)
        print("Evaluating best model on test set (no retraining)")
        print("=" * 60)
        evaluate_best_on_test(best_model, best_cfg, df_train, df_val, factor_list, norm_stats, exp_save_dir)


def list_experiments(experiments: list, hpo_enabled: bool) -> None:
    print("\n" + "=" * 70)
    print("Available experiments:")
    print("=" * 70)
    for i, exp in enumerate(experiments, 1):
        model_info = ", ".join(f'{m["type"]}({m.get("name", m["type"])})' for m in exp["models"])
        exp_name = exp.get("output", {}).get("experiment_name", exp.get("name", f"experiment_{i}"))
        mode_label = "HPO" if hpo_enabled else "Train"
        print(f"  [{i}] {exp_name}  —  models: {model_info}  ({mode_label})")
    print("=" * 70 + "\n")


def run_experiment(config_path: str, experiment_index: int) -> None:
    all_configs = load_configs(config_path)
    shared_data = all_configs["shared_data"]
    shared_hpo = all_configs.get("shared_hpo", {})
    hpo_enabled = shared_hpo.get("enabled", False)
    experiments = all_configs["experiments"]

    if experiment_index < 1 or experiment_index > len(experiments):
        print(f"Error: experiment index {experiment_index} out of range [1, {len(experiments)}]")
        sys.exit(1)

    experiment = experiments[experiment_index - 1]
    exp_name = experiment.get("name", f"experiment_{experiment_index}")

    mode_label = "HPO" if hpo_enabled else "Train"
    print("=" * 70)
    print(f"Running experiment [{experiment_index}/{len(experiments)}]: {exp_name}  ({mode_label})")
    print("=" * 70)

    cfg = build_config(shared_data, experiment)

    print(f"[Data] date_range: {shared_data['date_range']}")
    print(f"[Data] train: {cfg.data.train_dates}, val: {cfg.data.val_dates}, test: {cfg.data.test_dates}")
    print(f"[Models] {experiment['models']}")
    print(f"[Output] experiment_name: {cfg.output.experiment_name}")
    print(f"[Mode] {mode_label}")

    if hpo_enabled:
        print(f"[HPO] method={shared_hpo.get('method')}, n_trials={shared_hpo.get('n_trials')}, "
              f"metric={shared_hpo.get('metric')}, direction={shared_hpo.get('direction')}")
        run_hpo(cfg, shared_hpo)
        hpo_save_dir = os.path.join(shared_hpo.get("save_dir", "hpo_output"), cfg.output.experiment_name)
        upload_dir_to_hdfs(hpo_save_dir, cfg, hdfs_subdir="hpo_output")
    else:
        trainer = Trainer(cfg)
        trainer.run()

    output_dir = os.path.join(cfg.output.save_dir, cfg.output.experiment_name)
    upload_dir_to_hdfs(output_dir, cfg)

    print(f"\nExperiment [{experiment_index}] '{exp_name}' completed.\n")


def main():
    parser = argparse.ArgumentParser(description="Unified training runner for multiple experiments")
    parser.add_argument("--config", type=str, default="train_configs.yaml",
                        help="Path to the YAML config file (default: train_configs.yaml)")
    parser.add_argument("--id", type=int, nargs="*", default=None,
                        help="Experiment ID(s) to run (1-based). "
                             "If omitted, uses SLURM_ARRAY_TASK_ID or enters interactive mode.")
    args = parser.parse_args()

    all_configs = load_configs(args.config)
    shared_hpo = all_configs.get("shared_hpo", {})
    experiments = all_configs["experiments"]
    hpo_enabled = shared_hpo.get("enabled", False)

    if args.id is not None:
        for exp_id in args.id:
            run_experiment(args.config, exp_id)
    elif "SLURM_ARRAY_TASK_ID" in os.environ:
        exp_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        run_experiment(args.config, exp_id)
    else:
        list_experiments(experiments, hpo_enabled)
        try:
            choice = input("Enter experiment ID(s) to run (e.g. 1 or 1 2 3): ").strip()
            ids = [int(x) for x in choice.split()]
            for exp_id in ids:
                run_experiment(args.config, exp_id)
        except (ValueError, KeyboardInterrupt):
            print("\nAborted.")
            sys.exit(0)


if __name__ == "__main__":
    main()
