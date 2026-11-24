from pathlib import Path

import yaml

from pydantic import BaseModel, ConfigDict


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: Path
    target: str

class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    artifact_path: Path

class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metrics_path: Path

class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data: DataConfig
    model: ModelConfig
    evaluation: EvaluationConfig

def load_config(path: Path) -> PipelineConfig:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return PipelineConfig.model_validate(config)