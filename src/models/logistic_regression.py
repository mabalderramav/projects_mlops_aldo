from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.models.base import SklearnModel
from src.preprocessing.transformers import build_lr_preprocessor


class LogisticRegressionModel(SklearnModel):
    """Logistic Regression implementation."""

    def _build_model(self, **params: Any) -> Pipeline:

        return Pipeline([
            ("preprocessor", build_lr_preprocessor()),
            ("classifier", LogisticRegression(**params))
        ])