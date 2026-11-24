import json
import logging

from pathlib import Path
from typing import Any
from pydantic import BaseModel, ConfigDict

from src.training.config import ModelConfig


logger = logging.getLogger(__name__)

class TunedHyperparametersArtifact(BaseModel):
    """Contract for the tuning artifact."""

    model_config = ConfigDict(extra="ignore")
    model: str
    hyperparameters: dict[str, Any]


class HyperparameterResolver:
    """Resolve model hyperparameters from different sources."""

    def resolve(self, config: ModelConfig) -> dict[str, Any]:
        logger.info(
            "Resolving hyperparameters | model=%s | source=%s",
            config.name,
            config.hyperparameters_source,
        )

        if config.hyperparameters_source == "manual":
            if config.hyperparameters is None:
                raise ValueError("Manual hyperparameters not provided")
            return config.hyperparameters.copy()

        if config.hyperparameters_source == "tuning":
            if config.hyperparameters_path is None:
                raise ValueError("Tuning artifact path not provided")
            return self._load_tuned_hyperparameters(path=config.hyperparameters_path, expected_model=config.name)

        raise ValueError(f"Unsupported hyperparameters source: {config.hyperparameters_source}")

    def _load_tuned_hyperparameters(self, path: Path, expected_model: str) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"Tuning artifact not found: {path}")

        logger.info("Loading tuned hyperparameters: %s", path)

        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        artifact = TunedHyperparametersArtifact.model_validate(data)

        if artifact.model != expected_model:
            raise ValueError(f"Model mismatch in tuning artifact | expected={expected_model} | actual={artifact.model}")

        logger.info(
            "Tuned hyperparameters loaded | model=%s | parameters=%s",
            artifact.model,
            len(artifact.hyperparameters),
        )

        return artifact.hyperparameters.copy()