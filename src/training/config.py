from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ConfigDict, model_validator

from src.validation.config import DataConfig


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: Path
    target: str

class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    hyperparameters_source: Literal["manual", "tuning"] = "manual"
    hyperparameters: dict[str, Any] | None = None
    hyperparameters_path: Path | None = None

    @model_validator(mode="after")
    def validate_hyperparameters_source(self):
        if self.hyperparameters_source == "manual":
            if self.hyperparameters is None:
                raise ValueError("Manual training requires hyperparameters")
            if self.hyperparameters_path is not None:
                raise ValueError("Manual training does not require hyperparameters_path")

        elif self.hyperparameters_source == "tuning":
            if self.hyperparameters_path is None:
                raise ValueError("Tuned training requires hyperparameters_path")
            if self.hyperparameters is not None:
                raise ValueError("Tuned training does not accept manual hyperparameters")
        return self

class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    artifact_path: Path

class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig

def load_config(path: Path) -> PipelineConfig:
    """Load and validate the training configuration."""
    
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return PipelineConfig.model_validate(config)