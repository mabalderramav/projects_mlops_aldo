from typing import Any

from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

from src.models.base import SklearnModel
from src.preprocessing.transformers import build_tree_preprocessor


class XGBoostModel(SklearnModel):
    """XGBoost implementation."""

    def _build_model(self, **params: Any) -> Pipeline:
        return Pipeline([
            ("preprocessor", build_tree_preprocessor()),
            ("classifier", XGBClassifier(**params))
        ])