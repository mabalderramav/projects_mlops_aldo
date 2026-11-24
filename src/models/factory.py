import logging
from typing import Any
from pathlib import Path

from src.models.base import BaseModel
from src.models.logistic_regression import LogisticRegressionModel
from src.models.random_forest import RandomForestModel
from src.models.xgboost import XGBoostModel


logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory for creating classification models."""

    _registry: dict[str, type[BaseModel]] = {
        "logistic_regression": LogisticRegressionModel,
        "random_forest": RandomForestModel,
        "xgboost": XGBoostModel,
    }

    @classmethod
    def create(cls, name: str, **params: Any) -> BaseModel:
        
        if name not in cls._registry:
            available = ", ".join(cls._registry)

            raise ValueError(
                f"Unsupported model: {name}. "
                f"Available models: {available}"
            )

        logger.info("Creating model: %s", name)
        model_class = cls._registry[name]
        return model_class(**params)

    @classmethod
    def available_models(cls) -> list[str]:
        return list(cls._registry.keys())

    @classmethod
    def load(cls, name: str, path: Path) -> BaseModel:
        if name not in cls._registry:
            raise ValueError(
                f"Unsupported model: {name}"
            )

        if not path.is_file():
            raise FileNotFoundError(
                f"Model artifact not found: {path}"
            )

        logger.info(
            "Loading model | name=%s | path=%s",
            name,
            path,
        )

        model_class = cls._registry[name]
        return model_class.load(path)