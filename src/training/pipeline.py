import logging

from pathlib import Path

from src.data.loader import load_data, split_features_target
from src.models.factory import ModelFactory
from src.training.config import load_config
from src.training.hyperparameters import HyperparameterResolver
from src.training.trainer import Trainer


logger = logging.getLogger(__name__)

class TrainingPipeline:
    """Orchestrate the model training process."""

    def __init__(self, config_path: Path):
        self.config_path = config_path

    def run(self) -> None:
        logger.info("Training pipeline started")

        # 1. Load configuration
        config = load_config(self.config_path)

        # 2. Load prepared data
        df = load_data(config.data.path)

        # 3. Separate features and target
        X_train, y_train = split_features_target(df, config.data.target)

        # 4. Resolve hyperparameters
        resolver = HyperparameterResolver()
        hyperparameters = resolver.resolve(config.model)

        # 5. Create model
        model = ModelFactory.create(
            name=config.model.name,
            **hyperparameters,
        )

        # 6. Train model
        trainer = Trainer(model)
        trainer.train(X_train, y_train)

        # 7. Save model artifact
        trainer.save(config.training.artifact_path)

        logger.info("Hyperparameters used for %s: %s", config.model.name, hyperparameters)
        logger.info("Training pipeline completed")