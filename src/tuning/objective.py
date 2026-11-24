import logging

import optuna
import pandas as pd

from src.models.factory import ModelFactory

from src.tuning.config import TuningConfig
from src.tuning.search_space import SearchSpace
from src.validation.validator import CrossValidationValidator


logger = logging.getLogger(__name__)

class Objective:
    """Evaluate a candidate hyperparameter configuration."""

    def __init__(self, config: TuningConfig, validator: CrossValidationValidator, X: pd.DataFrame, y: pd.Series) -> None:
        self.config = config
        self.validator = validator
        self.X = X
        self.y = y
        self.search_space = SearchSpace(config=config.model.search_space)

    def __call__(self, trial: optuna.Trial) -> float:

        # 1. Generate candidate hyperparameters
        suggested_params = self.search_space.suggest(trial)

        # 2. Merge fixed and suggested hyperparameters
        hyperparameters = {
            **self.config.model.fixed_hyperparameters,
            **suggested_params
        }

        logger.info("Trial started | trial=%s | hyperparameters=%s", trial.number, hyperparameters)

        # 3. Create candidate model
        model = ModelFactory.create(name=self.config.model.name, **hyperparameters)

        # 4. Validate candidate model
        results = self.validator.validate(estimator=model.get_estimator(), X=self.X, y=self.y)

        # 5. Retrieve objective score
        metric = self.config.tuning.objective_metric

        score = float(results["metrics"][metric]["mean"])

        # 6. Store additional trial information
        trial.set_user_attr("hyperparameters", hyperparameters)
        trial.set_user_attr("validation_results", results)

        logger.info(
            "Trial completed | trial=%s | metric=%s | score=%.4f",
            trial.number,
            metric,
            score,
        )

        return score