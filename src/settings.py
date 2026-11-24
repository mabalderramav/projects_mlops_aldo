from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    app_env: Literal["development", "testing", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    training_config_path: Path = Path("configs/training.yaml")
    evaluation_config_path: Path = Path("configs/evaluation.yaml")
    data_config_path: Path = Path("configs/data.yaml")
    validation_config_path: Path = Path("configs/validation.yaml")
    tuning_config_path: Path = Path("configs/tuning.yaml")

    model_config = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    extra="ignore",
)