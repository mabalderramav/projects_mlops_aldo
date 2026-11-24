from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.validation.config import DataConfig, CrossValidationConfig, MetricName


class IntSearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["int"]
    low: int
    high: int
    log: bool = False

    @model_validator(mode="after")
    def validate_bounds(self):
        if self.low > self.high:
            raise ValueError("low must be <= high")
        if self.log and self.low <= 0:
            raise ValueError("Logarithmic search requires low > 0")
        return self

class FloatSearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["float"]
    low: float
    high: float
    log: bool = False

    @model_validator(mode="after")
    def validate_bounds(self):
        if self.low > self.high:
            raise ValueError("low must be <= high")
        if self.log and self.low <= 0:
            raise ValueError("Logarithmic search requires low > 0")
        return self


class CategoricalSearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["categorical"]
    choices: list[str | int | float | bool | None] = Field(min_length=1)

SearchParameterConfig = Annotated[IntSearchConfig | FloatSearchConfig | CategoricalSearchConfig, Field(discriminator="type")]

class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    fixed_hyperparameters: dict[str, Any] = Field(default_factory=dict)
    baseline_hyperparameters: dict[str, Any] = Field(default_factory=dict)
    search_space: dict[str, SearchParameterConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_hyperparameters(self):
        overlap = (set(self.fixed_hyperparameters) & set(self.search_space))

        if overlap:
            raise ValueError(f"Fixed parameters cannot be tuned: {sorted(overlap)}")

        unknown = (set(self.baseline_hyperparameters) - set(self.search_space))
        if unknown:
            raise ValueError(f"Baseline parameters outside search space: {sorted(unknown)}")
        return self

class OptimizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_trials: int = Field(default=50, ge=1)
    direction: Literal["maximize", "minimize"] = "maximize"
    objective_metric: MetricName = "average_precision"
    random_state: int = 42
    n_jobs: int = 1
    include_baseline: bool = True

class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    best_hyperparameters_path: Path
    results_path: Path

class TuningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data: DataConfig
    model: ModelConfig
    validation: CrossValidationConfig
    tuning: OptimizationConfig
    output: OutputConfig

    @model_validator(mode="after")
    def validate_configuration(self):
        if self.tuning.objective_metric not in self.validation.metrics:
            raise ValueError("Objective metric must be included in validation metrics")

        if self.tuning.include_baseline:
            missing = (set(self.model.search_space) - set(self.model.baseline_hyperparameters))

            if missing:
                raise ValueError(f"Missing baseline parameters: {sorted(missing)}")

            for name, space in self.model.search_space.items():
                value = self.model.baseline_hyperparameters[name]
                if isinstance(space, IntSearchConfig):
                    valid = (type(value) is int and space.low <= value <= space.high)
                elif isinstance(space, FloatSearchConfig):
                    valid = (type(value) in (int, float) and space.low <= value <= space.high)
                else:
                    valid = value in space.choices

                if not valid:
                    raise ValueError(f"Invalid baseline value for {name}: {value}")

        if self.output.best_hyperparameters_path.resolve() == self.output.results_path.resolve():
            raise ValueError("Output artifact paths must be different")

        return self


def load_config(config_path: Path) -> TuningConfig:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return TuningConfig.model_validate(config)