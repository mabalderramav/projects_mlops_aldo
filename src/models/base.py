from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class BaseModel(ABC):
    """Abstract contract for classification models."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def get_estimator(self) -> BaseEstimator:
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseModel":
        pass


class SklearnModel(BaseModel):
    """Base implementation for sklearn-compatible models."""

    def __init__(self, **params: Any):
        self.params = params
        self.model = self._build_model(**params)

    @abstractmethod
    def _build_model(self, **params: Any):
        """Build the concrete estimator."""
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_estimator(self) -> BaseEstimator:
        return self.model

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "SklearnModel":
        model = joblib.load(path)

        if not isinstance(model, cls):
            raise TypeError(
                f"Invalid model artifact: {path}"
            )

        return model