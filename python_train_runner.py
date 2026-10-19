from model_trainer.config import Config, ModelEntry, DataConfig, TrainingConfig, OutputConfig
from model_trainer.trainer import Trainer

# 方式一：从 YAML 文件加载
# cfg = Config.from_yaml("configs/my_run.yaml")

# 方式二：代码构建
cfg = Config.from_dict({
    "data": {
        "data_root": "../FactorData",
        "train_dates": "20241024-20241104",
        "val_dates":   "20241105-20241107",
        "test_dates":  "20241108-20241112",
        "factor_list": ["p_288","p_291","p_294","p_297","p_303"],
        "y_list":      ["p_288_target"],
        "normalize":   "rolling",
    },
    "models": [
        {"type": "lgbm", "name": "lgbm", "params": {}},
        {"type": "mlp",  "name": "mlp",  "params": {"hidden_sizes": [128, 64]}},
    ],
    "training": {"epochs": 200, "batch_size": 4096, "early_stopping_patience": 10},
    "output":   {"save_dir": "output", "experiment_name": "test"},
})

# 运行实验
trainer = Trainer(cfg)
summary_df = trainer.run()   # 返回 summary DataFrame
