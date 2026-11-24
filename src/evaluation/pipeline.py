import json
import logging

from pathlib import Path

from src.data.loader import load_data, split_features_target
from src.models.factory import ModelFactory
from src.evaluation.config import load_config
from src.evaluation.evaluator import Evaluator


logger = logging.getLogger(__name__)

class EvaluationPipeline:
    """Orchestrate model evaluation."""

    def __init__(self, config_path: Path):
        self.config_path = config_path

    def run(self) -> dict:
        logger.info("Evaluation pipeline started")

        # 1. Load configuration
        config = load_config(self.config_path)

        # 2. Load test dataset
        df = load_data(config.data.path)

        # 3. Separate features and target
        X_test, y_test = split_features_target(
            df,
            config.data.target,
        )

        # 4. Load trained model
        model = ModelFactory.load(
            name=config.model.name,
            path=config.model.artifact_path,
        )

        # 5. Evaluate model
        evaluator = Evaluator(model)
        metrics = evaluator.evaluate(X_test, y_test)

        # 6. Save evaluation results
        metrics_path = config.evaluation.metrics_path
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=4)

        logger.info(
            "Evaluation metrics saved: %s",
            metrics_path,
        )
        logger.info("Evaluation pipeline completed")

        return metrics