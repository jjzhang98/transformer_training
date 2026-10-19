"""Generate independent python_train_runner_XXXXXX.py files from train_configs.yaml.

Usage:
    python generate_runners.py                          # use default train_configs.yaml
    python generate_runners.py --config my_configs.yaml # use custom config
    python generate_runners.py --output-dir ./runners   # output to custom directory

Generated files are self-contained and can be run independently,
which is useful for SLURM array jobs or manual execution.
"""

import argparse
import os
import pprint

import yaml


def split_date_range_code() -> str:
    return '''def split_date_range(date_range: str, train_ratio: float = 0.5, val_ratio: float = 0.25) -> tuple:
    start, end = date_range.split("-")
    dates = pd.date_range(start=start, end=end, freq="D")
    n_dates = len(dates)

    n_train = int(n_dates * train_ratio)
    n_val = int(n_dates * val_ratio)

    train_list = dates[:n_train].strftime("%Y%m%d").tolist()
    val_list = dates[n_train:n_train + n_val].strftime("%Y%m%d").tolist()
    test_list = dates[n_train + n_val:].strftime("%Y%m%d").tolist()

    return f"{train_list[0]}-{train_list[-1]}", f"{val_list[0]}-{val_list[-1]}", f"{test_list[0]}-{test_list[-1]}"
'''


def _hpo_evaluate_code() -> str:
    return '''def _evaluate_best_on_test(best_model, best_cfg, df_train, df_val, factor_list, norm_stats, hpo_save_dir):
    from model_trainer.data import load_split
    from model_trainer.metrics import evaluate, build_prior_topk_abtypes

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
'''


def generate_runner_script(shared_data: dict, shared_hpo: dict, experiment: dict, experiment_index: int) -> str:
    date_range = shared_data["date_range"]
    train_ratio = shared_data.get("train_ratio", 0.5)
    val_ratio = shared_data.get("val_ratio", 0.25)
    exp_name = experiment.get("name", f"experiment_{experiment_index}")

    _TRAIN_DATES_PLACEHOLDER = "__TRAIN_DATES_VAR__"
    _VAL_DATES_PLACEHOLDER = "__VAL_DATES_VAR__"
    _TEST_DATES_PLACEHOLDER = "__TEST_DATES_VAR__"

    data_dict = {
        "data_root": shared_data["data_root"],
        "train_dates": _TRAIN_DATES_PLACEHOLDER,
        "val_dates": _VAL_DATES_PLACEHOLDER,
        "test_dates": _TEST_DATES_PLACEHOLDER,
        "factor_list": shared_data.get("factor_list"),
        "y_list": shared_data.get("y_list"),
        "normalize": shared_data.get("normalize", "none"),
        "winsorize_std": shared_data.get("winsorize_std", 0),
    }
    if shared_data.get("abtypes") is not None:
        data_dict["abtypes"] = shared_data["abtypes"]
    if shared_data.get("codes") is not None:
        data_dict["codes"] = shared_data["codes"]

    config_dict = {
        "data": data_dict,
        "models": experiment["models"],
        "training": experiment.get("training", {}),
        "output": experiment.get("output", {}),
    }
    if "filter" in experiment:
        config_dict["filter"] = experiment["filter"]

    config_str = pprint.pformat(config_dict, width=120, sort_dicts=False)
    config_str = config_str.replace(f"'{_TRAIN_DATES_PLACEHOLDER}'", "train_dates")
    config_str = config_str.replace(f"'{_VAL_DATES_PLACEHOLDER}'", "val_dates")
    config_str = config_str.replace(f"'{_TEST_DATES_PLACEHOLDER}'", "test_dates")

    hpo_enabled = experiment.get("hpo", False)

    if hpo_enabled:
        hpo_shared_str = pprint.pformat(shared_hpo, width=120, sort_dicts=False)
        script = f'''import os
import pandas as pd
from model_trainer.config import Config
from model_trainer.trainer import Trainer
from model_trainer.tuner import HyperparamTuner, get_default_search_space


{split_date_range_code()}

{_hpo_evaluate_code()}

date_range = "{date_range}"
train_dates, val_dates, test_dates = split_date_range(date_range, train_ratio={train_ratio}, val_ratio={val_ratio})
print(f"[自动分割] train: {{train_dates}}, val: {{val_dates}}, test: {{test_dates}}")

cfg = Config.from_dict({config_str})

SHARED_HPO = {hpo_shared_str}

if __name__ == "__main__":
    method = SHARED_HPO.get("method", "random")
    n_trials = SHARED_HPO.get("n_trials", 50)
    metric = SHARED_HPO.get("metric", "auto")
    direction = SHARED_HPO.get("direction", "maximize")
    seed = SHARED_HPO.get("seed", 42)
    save_dir = os.path.join(SHARED_HPO.get("save_dir", "hpo_output"), cfg.output.experiment_name)
    disable_torch_compile = SHARED_HPO.get("disable_torch_compile", False)
    optuna_startup_trials = SHARED_HPO.get("optuna_startup_trials", 10)

    model_type = cfg.models[0].type
    search_space = get_default_search_space(model_type)

    print(f"Search space for '{{model_type}}':")
    for section, params in search_space.items():
        print(f"  [{{section}}]")
        for k, spec in params.items():
            print(f"    {{k}}: {{spec}}")

    tuner = HyperparamTuner(
        base_config=cfg,
        search_space=search_space,
        method=method,
        n_trials=n_trials,
        metric=metric,
        direction=direction,
        seed=seed,
        save_dir=save_dir,
        disable_torch_compile=disable_torch_compile,
        optuna_startup_trials=optuna_startup_trials,
    )

    result = tuner.run()

    best_cfg = result["best_config"]
    best_model = result["best_model"]
    df_train = result["df_train"]
    df_val = result["df_val"]
    factor_list = result["factor_list"]
    norm_stats = result["norm_stats"]

    if best_model is not None and best_cfg is not None:
        print("Evaluating best model on test set (no retraining)")
        _evaluate_best_on_test(best_model, best_cfg, df_train, df_val, factor_list, norm_stats, save_dir)

    def _upload_dir_to_hdfs(local_dir, cfg, hdfs_subdir=None):
        if not cfg.output.upload_to_hdfs:
            return
        if not os.path.isdir(local_dir):
            return
        try:
            from pyarrow import fs as pa_fs
            os.environ["HADOOP_USER_NAME"] = cfg.output.hdfs_user
            hdfs_fs = pa_fs.HadoopFileSystem.from_uri(cfg.output.hdfs_uri)
            if hdfs_subdir:
                remote_dir = os.path.join(cfg.output.hdfs_remote_root, hdfs_subdir, os.path.basename(local_dir))
            else:
                remote_dir = os.path.join(cfg.output.hdfs_remote_root, os.path.basename(local_dir))
            print(f"[Upload] Uploading to HDFS: {{local_dir}} -> {{remote_dir}}")
            hdfs_fs.create_dir(remote_dir, recursive=True)
            pa_fs.copy_files(source=local_dir, destination=remote_dir, source_filesystem=pa_fs.LocalFileSystem(), destination_filesystem=hdfs_fs)
            print(f"[Upload] Done: {{remote_dir}}")
        except Exception as exc:
            print(f"[Upload] Failed: {{exc}}")

    _upload_dir_to_hdfs(save_dir, cfg, hdfs_subdir="hpo_output")
    _upload_dir_to_hdfs(os.path.join(cfg.output.save_dir, cfg.output.experiment_name), cfg)
'''
    else:
        script = f'''import os
from model_trainer.config import Config
from model_trainer.trainer import Trainer
import pandas as pd


{split_date_range_code()}

date_range = "{date_range}"
train_dates, val_dates, test_dates = split_date_range(date_range, train_ratio={train_ratio}, val_ratio={val_ratio})
print(f"[自动分割] train: {{train_dates}}, val: {{val_dates}}, test: {{test_dates}}")

cfg = Config.from_dict({config_str})

if __name__ == "__main__":
    trainer = Trainer(cfg)
    summary_df = trainer.run()

    def _upload_dir_to_hdfs(local_dir, cfg):
        if not cfg.output.upload_to_hdfs:
            return
        if not os.path.isdir(local_dir):
            return
        try:
            from pyarrow import fs as pa_fs
            os.environ["HADOOP_USER_NAME"] = cfg.output.hdfs_user
            hdfs_fs = pa_fs.HadoopFileSystem.from_uri(cfg.output.hdfs_uri)
            remote_dir = os.path.join(cfg.output.hdfs_remote_root, os.path.basename(local_dir))
            print(f"[Upload] Uploading to HDFS: {{local_dir}} -> {{remote_dir}}")
            pa_fs.copy_files(source=local_dir, destination=remote_dir, source_filesystem=pa_fs.LocalFileSystem(), destination_filesystem=hdfs_fs)
            print(f"[Upload] Done: {{remote_dir}}")
        except Exception as exc:
            print(f"[Upload] Failed: {{exc}}")

    _upload_dir_to_hdfs(os.path.join(cfg.output.save_dir, cfg.output.experiment_name), cfg)
'''

    return script


def main():
    parser = argparse.ArgumentParser(description="Generate independent training runner scripts from YAML config")
    parser.add_argument("--config", type=str, default="train_configs.yaml",
                        help="Path to the YAML config file (default: train_configs.yaml)")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory to write generated scripts (default: current directory)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        all_configs = yaml.safe_load(f)

    shared_data = all_configs["shared_data"]
    shared_hpo = all_configs.get("shared_hpo", {})
    hpo_enabled = shared_hpo.get("enabled", False)
    experiments = all_configs["experiments"]

    os.makedirs(args.output_dir, exist_ok=True)

    generated = []
    for i, experiment in enumerate(experiments, 1):
        script_content = generate_runner_script(shared_data, shared_hpo, experiment, i)
        filename = f"python_train_runner_{i:06d}.py"
        filepath = os.path.join(args.output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(script_content)

        exp_name = experiment.get("name", f"experiment_{i}")
        hpo_tag = " [HPO]" if hpo_enabled else ""
        generated.append((filename, exp_name))
        print(f"  Generated: {filepath}  ({exp_name}{hpo_tag})")

    print(f"\nTotal: {len(generated)} runner scripts generated in '{args.output_dir}/'")
    print(f"SLURM array range: --array=1-{len(generated)}")


if __name__ == "__main__":
    main()
