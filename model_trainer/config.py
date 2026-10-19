"""Configuration dataclasses for the model training framework.

Usage:
    cfg = Config.from_yaml(path="configs/example.yaml")
    cfg = Config.from_dict(raw_dict={...})
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DataConfig:
    """Data source and preprocessing settings.

    Attributes:
        data_root: Root directory containing date-partitioned parquet files.
        train_dates: Training date range in "YYYYMMDD-YYYYMMDD" format.
        val_dates: Validation date range in "YYYYMMDD-YYYYMMDD" format.
        test_dates: Test date range in "YYYYMMDD-YYYYMMDD" format.
        y_list: List of label column names; one model is trained per label.
        factor_list: List of feature column names; None means auto-detect all signal_* columns.
        abtypes: Optional list of abtype codes to filter (e.g. [3018, 3232]); None uses all.
        codes: Optional list of instrument codes to filter (e.g. [12502, 22502]); None uses all.
        normalize: Normalization strategy — "rolling" | "global" | "group" | "none".
        winsorize_std: Clip z-scores to ±N after normalization; 0 disables clipping.
    """

    data_root: str
    train_dates: str
    val_dates: str
    test_dates: str
    y_list: List[str]
    factor_list: Optional[List[str]] = None
    abtypes: Optional[List[int]] = None
    codes: Optional[List[int]] = None
    normalize: str = "rolling"
    winsorize_std: float = 3.0


@dataclass
class TrainingConfig:
    """Training loop hyperparameters shared across all deep learning models.

    Attributes:
        epochs: Maximum number of training epochs.
        batch_size: Number of groups per training batch.
        learning_rate: Adam optimizer initial learning rate.
        early_stopping_patience: Epochs without improvement before stopping.
        min_epochs: Minimum epochs to train before early stopping can trigger.
        min_delta: Minimum improvement in validation metric to reset patience counter.
        device: Compute device — "auto" | "cpu" | "cuda" | "mps".
        loss_type: Loss function type — "pearson" | "listnet" | "listmle".
        topk: K value for top-k hit rate metric and top-k hard label construction.
        rank_before_softmax: Whether to rank-transform labels before softmax (for listnet loss).
        use_hard_label: If True, convert labels to top-k hard labels (1/k for top-k, 0 otherwise);
                        if False, use soft labels (rank + softmax).
        label_smoothing: Label smoothing coefficient in [0, 1]; 0 disables smoothing.
        torch_compile: Whether to use torch.compile for model and loss optimization (requires PyTorch 2.0+).
    """

    epochs: int = 200
    batch_size: int = 1024
    learning_rate: float = 3e-4
    early_stopping_patience: int = 30
    min_epochs: int = 30
    min_delta: float = 1e-4
    device: str = "auto"
    loss_type: str = "pearson"
    topk: int = 1
    rank_before_softmax: bool = True
    use_hard_label: bool = False
    label_smoothing: float = 0.0
    torch_compile: bool = True


@dataclass
class FilterConfig:
    """Lasso-based feature selection configuration.

    Attributes:
        alphas: List of L1 penalty strengths to search over.
        filter_dates: Date range for the feature selection data split.
        factor_list: Feature columns to consider; None uses all.
        output_dir: Directory to save feature selection results.
        data_root: Root directory for data files.
        y_list: Label columns for feature selection; None uses all.
        abtypes: Optional abtype filter for feature selection data.
        codes: Optional instrument code filter for feature selection data.
        normalize: Whether to normalize features before selection.
        winsorize_std: Z-score clipping threshold; None disables clipping.
        threshold: Coefficient magnitude below which features are considered zero.
    """

    alphas: List[float] = field(default_factory=lambda: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0])
    filter_dates: str = ""
    factor_list: Optional[List[str]] = None
    output_dir: str = "filter_output"
    data_root: str = ""
    y_list: Optional[List[str]] = None
    abtypes: Optional[List[str]] = None
    codes: Optional[List[int]] = None
    normalize: bool = True
    winsorize_std: Optional[float] = None
    threshold: float = 1e-3


@dataclass
class OutputConfig:
    """Output storage and HDFS upload configuration.

    Attributes:
        save_dir: Local directory to save experiment outputs.
        experiment_name: Unique name for this experiment run.
        save_model: Whether to persist model weights to disk.
        upload_to_hdfs: Whether to upload outputs to HDFS.
        hdfs_uri: HDFS filesystem URI.
        hdfs_user: HDFS authentication username.
        hdfs_remote_root: HDFS remote directory for uploaded outputs.
    """

    save_dir: str = "output"
    experiment_name: str = "exp"
    save_model: bool = True
    upload_to_hdfs: bool = True
    hdfs_uri: str = "viewfs://ftxz-hadoop-view/"
    hdfs_user: str = "zli"
    hdfs_remote_root: str = "/ext1/user/zli/zjj/model_trainer/modeltrainer/output"


@dataclass
class ModelEntry:
    """Specification for one model to train, including its type and hyperparameters.

    Attributes:
        type: Model architecture identifier — "lgbm" | "mlp" | "transformer" | "lambdamart".
        name: Display name for this model; defaults to type if empty.
        params: Dictionary of model-specific hyperparameters passed to the constructor.
    """

    type: str
    name: str = ""
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set default name to model type if not provided."""
        if not self.name:
            self.name = self.type


@dataclass
class Config:
    """Root configuration object that aggregates all sub-configurations.

    Attributes:
        data: Data source and preprocessing settings.
        models: List of model specifications to train.
        training: Training loop hyperparameters.
        output: Output storage and upload settings.
        filter: Optional feature selection configuration.
    """

    data: DataConfig
    models: List[ModelEntry]
    training: TrainingConfig
    output: OutputConfig
    filter: Optional[FilterConfig] = None

    @classmethod
    def from_yaml(cls, path: str) -> Config:
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Fully constructed Config object.
        """
        with open(path, "r") as yaml_file:
            raw_dict = yaml.safe_load(yaml_file)
        return cls.from_dict(raw_dict=raw_dict)

    @classmethod
    def from_dict(cls, raw_dict: dict) -> Config:
        """Construct Config from a raw dictionary (e.g. parsed YAML).

        Args:
            raw_dict: Dictionary with keys "data", "models", "training", "output", optionally "filter".

        Returns:
            Fully constructed Config object.
        """
        data_cfg = DataConfig(**raw_dict["data"])
        training_cfg = TrainingConfig(**raw_dict.get("training", {}))
        output_cfg = OutputConfig(**raw_dict.get("output", {}))
        filter_cfg = FilterConfig(**raw_dict.get("filter", {})) if "filter" in raw_dict else None
        model_entries = [
            ModelEntry(
                type=model_dict["type"],
                name=model_dict.get("name", model_dict["type"]),
                params=model_dict.get("params", {}),
            )
            for model_dict in raw_dict.get("models", [])
        ]
        return cls(
            data=data_cfg,
            models=model_entries,
            training=training_cfg,
            output=output_cfg,
            filter=filter_cfg,
        )
