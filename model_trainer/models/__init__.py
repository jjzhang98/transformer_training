"""Model registry and factory."""
from __future__ import annotations

from .base import BaseModel
from .lgbm import LGBMModel
from .lasso import LassoModel
from .mlp import MLPModel
from .transformer import TransformerModel
from .gru import GRUModel
from .dlinear import DLinearModel
from .lambdamart import LambdaMARTModel

_REGISTRY = {
    "lgbm":        LGBMModel,
    "lasso":       LassoModel,
    "mlp":         MLPModel,
    "transformer": TransformerModel,
    "gru":         GRUModel,
    "dlinear":     DLinearModel,
    "lambdamart":  LambdaMARTModel,
}


def build_model(model_type: str, name: str = "", **params) -> BaseModel:
    """Instantiate a model by type name.

    Args:
        model_type: "lgbm" | "lasso" | "mlp" | "transformer" | "gru" | "dlinear" | "lambdamart"
        name:       Human-readable identifier (used in output paths).
        **params:   Model-specific hyperparameters.

    Returns:
        Initialised (untrained) model instance.
    """
    key = model_type.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown model type '{model_type}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[key](name=name or model_type, **params)


__all__ = ["BaseModel", "LGBMModel", "LassoModel", "MLPModel", "TransformerModel", "GRUModel", "DLinearModel", "LambdaMARTModel", "build_model"]