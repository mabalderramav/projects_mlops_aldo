from pathlib import Path
from typing import Any, Literal

import yaml

from pydantic import BaseModel, ConfigDict, Field


MetricName = Literal["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]

class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: Path
    target: str

class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    hyperparameters: dict[str, Any] = Field(default_factory=dict)

class CrossValidationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strategy: Literal["stratified_kfold"] = "stratified_kfold"
    n_splits: int = Field(default=5, ge=2)
    shuffle: bool = True
    random_state: int = 42
    n_jobs: int = 1
    metrics: list[MetricName] = Field(default_factory=lambda: ["roc_auc", "average_precision", "f1"], min_length=1)

class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metrics_path: Path

class ValidationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data: DataConfig
    model: ModelConfig
    validation: CrossValidationConfig
    output: OutputConfig

def load_config(config_path: Path) -> ValidationConfig:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return ValidationConfig.model_validate(config)