from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, AwareDatetime


class SourcesConfig(BaseModel):

    model_config = ConfigDict(extra="forbid")

    orders: Path
    order_details: Path
    products: Path
    users: Path


class PreprocessingConfig(BaseModel):

    model_config = ConfigDict(extra="forbid")

    order_status: str = "Complete"


class TargetConfig(BaseModel):

    model_config = ConfigDict(extra="forbid")

    name: str = "repurchased_60d"
    horizon_days: int = Field(default=60, gt=0)
    observation_end: AwareDatetime


class SplitConfig(BaseModel):

    model_config = ConfigDict(extra="forbid")

    test_size: float = Field(default=0.2, gt=0, lt=1)
    random_state: int = 42
    stratify: bool = True


class OutputConfig(BaseModel):

    model_config = ConfigDict(extra="forbid")

    train_path: Path
    test_path: Path
    metadata_path: Path


class PipelineConfig(BaseModel):

    model_config = ConfigDict(extra="forbid")

    sources: SourcesConfig
    preprocessing: PreprocessingConfig
    target: TargetConfig
    split: SplitConfig
    output: OutputConfig


def load_config(path: Path) -> PipelineConfig:

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return PipelineConfig.model_validate(config)