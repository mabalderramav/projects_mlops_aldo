import json
import logging

from pathlib import Path

import optuna
from optuna.trial import TrialState

from src.data.loader import load_data, split_features_target
from src.tuning.config import load_config
from src.tuning.objective import Objective
from src.tuning.tuner import HyperparameterTuner
from src.validation.validator import CrossValidationValidator


logger = logging.getLogger(__name__)

class TuningPipeline:
    """Orchestrate hyperparameter optimization."""

    def __init__(self, config_path: Path):
        self.config_path = config_path

    def run(self) -> dict:
        logger.info("Tuning pipeline started")

        # 1. Load configuration
        config = load_config(self.config_path)

        # 2. Load development dataset
        df = load_data(config.data.path)

        # 3. Separate features and target
        X, y = split_features_target(df, config.data.target)

        # 4. Create cross-validation validator
        validator = CrossValidationValidator(config=config.validation)

        # 5. Create objective
        objective = Objective(config=config, validator=validator, X=X, y=y)

        # 6. Create tuner
        tuner = HyperparameterTuner(config=config.tuning)

        # 7. Execute hyperparameter optimization
        study = tuner.tune(objective=objective, baseline_params=config.model.baseline_hyperparameters)

        # 8. Build best hyperparameters artifact
        best_trial = study.best_trial

        best_hyperparameters = {
            "model": config.model.name,
            "objective_metric": config.tuning.objective_metric,
            "direction": config.tuning.direction,
            "best_trial": best_trial.number,
            "best_score": float(study.best_value),
            "n_trials": len(study.trials),
            "n_splits": config.validation.n_splits,
            "hyperparameters": best_trial.user_attrs["hyperparameters"],
            "validation_metrics": best_trial.user_attrs["validation_results"]["metrics"]
        }

        # 9. Build study results
        tuning_results = self._build_results(study)

        tuning_results.update({
            "model": config.model.name,
            "dataset": str(config.data.path),
            "target_column": config.data.target,
            "n_samples": len(X),
            "positive_rate": float(y.mean()),
            "objective_metric": config.tuning.objective_metric,
            "direction": config.tuning.direction,
            "strategy": config.validation.strategy,
            "n_splits": config.validation.n_splits,
            "configuration": config.model_dump(mode="json")
        })

        # 10. Save artifacts
        self._save_json(config.output.best_hyperparameters_path, best_hyperparameters)
        self._save_json(config.output.results_path, tuning_results)
        logger.info("Tuning pipeline completed")
        return best_hyperparameters

    def _build_results(self, study: optuna.Study) -> dict:
        trials = []

        for trial in study.trials:
            trials.append({
                "number": trial.number,
                "state": trial.state.name,
                "score": trial.value,
                "suggested_hyperparameters": trial.params,
                "hyperparameters": trial.user_attrs.get("hyperparameters"),
                "is_baseline": trial.user_attrs.get("is_baseline", False),
                "validation_results": trial.user_attrs.get("validation_results"),
                "duration_seconds": (trial.duration.total_seconds() if trial.duration is not None else None)
            })

        baseline_trial = next(
            (
                trial for trial in study.trials
                if (
                    trial.user_attrs.get("is_baseline", False)
                    and trial.state == TrialState.COMPLETE
                )
            ), None
        )

        baseline_score = float(baseline_trial.value) if baseline_trial is not None else None

        return {
            "best_trial": study.best_trial.number,
            "best_score": float(study.best_value),
            "baseline_score": baseline_score,
            "n_trials": len(study.trials),
            "completed_trials": sum(trial.state == TrialState.COMPLETE for trial in study.trials),
            "failed_trials": sum(trial.state == TrialState.FAIL for trial in study.trials),
            "pruned_trials": sum(trial.state == TrialState.PRUNED for trial in study.trials),
            "trials": trials
        }

    def _save_json(self, path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, allow_nan=False)

        logger.info("Tuning artifact saved: %s", path)