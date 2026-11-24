import logging
from collections.abc import Callable

import optuna

from optuna.trial import TrialState

from src.tuning.config import OptimizationConfig


logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Manage the hyperparameter optimization study."""

    def __init__(self, config: OptimizationConfig) -> None:
        self.config = config

    def tune(self, objective: Callable[[optuna.Trial], float], baseline_params: dict | None = None) -> optuna.Study:

        logger.info(
            "Hyperparameter tuning started | trials=%s | metric=%s | direction=%s",
            self.config.n_trials,
            self.config.objective_metric,
            self.config.direction,
        )

        # 1. Create sampler
        sampler = optuna.samplers.TPESampler(seed=self.config.random_state)

        # 2. Create optimization study
        study = optuna.create_study(direction=self.config.direction, sampler=sampler)

        # 3. Include baseline configuration
        if self.config.include_baseline:
            if baseline_params is None:
                raise ValueError("Baseline hyperparameters are required")

            study.enqueue_trial(params=baseline_params, user_attrs={"is_baseline": True})

            logger.info("Baseline trial enqueued")

        # 4. Execute optimization
        study.optimize(objective, n_trials=self.config.n_trials, n_jobs=self.config.n_jobs,)

        # 5. Validate study results
        completed_trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]

        if not completed_trials:
            raise RuntimeError("No completed tuning trials")

        logger.info(
            "Hyperparameter tuning completed | best_trial=%s | best_score=%.4f",
            study.best_trial.number,
            study.best_value,
        )

        return study