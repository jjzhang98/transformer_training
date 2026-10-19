"""Abstract base class that all models must implement."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Interface shared by all model implementations.

    Design contract:
        - ``fit``    trains the model and returns a history dict.
        - ``predict`` returns predictions aligned row-by-row with the input df.
        - ``feature_importance`` is optional (returns None if not supported).
        - ``save`` / ``load`` handle model persistence.

    The ``fit`` and ``predict`` methods take a DataFrame (with meta + factors + label
    columns) so that temporal models can use group structure (date / code / abtype).
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(
        self,
        df_train: pd.DataFrame,
        factor_list: List[str],
        y_col: str,
        df_val: Optional[pd.DataFrame] = None,
        training_cfg=None,          # TrainingConfig dataclass or None
    ) -> dict:
        """Train the model.

        Returns:
            history dict, e.g. {"train_loss": [...], "val_ic": [...]}
        """

    @abstractmethod
    def predict(self, df: pd.DataFrame, factor_list: List[str]) -> np.ndarray:
        """Return predictions (1-D array) aligned with ``df`` rows.

        Rows the model cannot predict (e.g. insufficient history for a
        temporal model) should have NaN in the returned array.
        """

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    def feature_importance(self) -> Optional[pd.Series]:
        """Return a pd.Series of feature importances, or None."""
        return None

    def save(self, path: str) -> None:  # noqa: D401
        """Persist the model to ``path``."""

    def load(self, path: str) -> None:
        """Restore model weights from ``path``."""
