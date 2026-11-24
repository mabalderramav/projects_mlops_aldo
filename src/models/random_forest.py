from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.models.base import SklearnModel
from src.preprocessing.transformers import build_tree_preprocessor

class RandomForestModel(SklearnModel):
    """Random Forest implementation."""

    def _build_model(self, **params: Any) -> Pipeline:
        return Pipeline([
            ("preprocessor", build_tree_preprocessor()),
            ("classifier", RandomForestClassifier(**params))
        ])