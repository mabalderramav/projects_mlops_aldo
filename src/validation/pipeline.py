import json
import logging

from pathlib import Path

from src.data.loader import load_data, split_features_target
from src.models.factory import ModelFactory
from src.validation.config import load_config
from src.validation.validator import CrossValidationValidator


logger = logging.getLogger(__name__)

class ValidationPipeline:
    """Orchestrate model cross-validation."""

    def __init__(self, config_path: Path):
        self.config_path = config_path

    def run(self) -> dict:
        logger.info("Validation pipeline started")

        # 1. Load configuration
        config = load_config(self.config_path)

        # 2. Load development dataset
        df = load_data(config.data.path)

        # 3. Separate features and target
        X, y = split_features_target(df, config.data.target)

        # 4. Create model
        model = ModelFactory.create(name=config.model.name, **config.model.hyperparameters)

        # 5. Cross-validation
        validator = CrossValidationValidator(config=config.validation)

        results = validator.validate(estimator=model.get_estimator(), X=X, y=y)

        # 6. Build validation results
        metrics = {
            "model": config.model.name,
            "hyperparameters": config.model.hyperparameters,
            "strategy": config.validation.strategy,
            "dataset": str(config.data.path),
            "target_column": config.data.target,
            "positive_rate": float(y.mean()),
            **results,
        }

        # 7. Save validation results
        metrics_path = config.output.metrics_path
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=4, allow_nan=False)

        logger.info("Validation metrics saved: %s", metrics_path)
        logger.info("Validation pipeline completed")
        return metrics