import logging

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.validation.config import CrossValidationConfig


logger = logging.getLogger(__name__)

class CrossValidationValidator:
    def __init__(self, config: CrossValidationConfig) -> None:
        self.config = config
        self.cv = StratifiedKFold(
            n_splits=config.n_splits,
            shuffle=config.shuffle,
            random_state=(config.random_state if config.shuffle else None),
        )

    def _build_scoring(self) -> dict:
        available_scoring = {
            "accuracy": "accuracy",
            "precision": make_scorer(precision_score, zero_division=0),
            "recall": make_scorer(recall_score, zero_division=0),
            "f1": make_scorer(f1_score, zero_division=0),
            "roc_auc": "roc_auc",
            "average_precision": "average_precision",
        }

        return {
            metric: available_scoring[metric] for metric in self.config.metrics
        }

    def validate(self, estimator: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> dict:
        logger.info(
            "Cross-validation started | model=%s | folds=%s | samples=%s",
            estimator.__class__.__name__,
            self.config.n_splits,
            len(X),
        )

        class_counts = y.value_counts()

        if y.isna().any():
            raise ValueError("Target contains null values")

        if len(class_counts) != 2:
            raise ValueError("Binary classification requires two classes")

        if class_counts.min() < self.config.n_splits:
            raise ValueError("Insufficient samples in minority class for cross-validation")

        results = cross_validate(
            estimator=estimator,
            X=X,
            y=y,
            cv=self.cv,
            scoring=self._build_scoring(),
            n_jobs=self.config.n_jobs,
            return_train_score=False,
            error_score="raise",
        )

        metrics = self._aggregate_metrics(results)

        for metric, values in metrics.items():
            logger.info(
                "Validation metric | metric=%s | mean=%.4f | std=%.4f",
                metric,
                values["mean"],
                values["std"],
            )

        logger.info("Cross-validation completed")

        return {
            "n_splits": self.config.n_splits,
            "n_samples": len(X),
            "metrics": metrics,
            "timing": {
                "fit_time_mean": float(results["fit_time"].mean()),
                "score_time_mean": float(results["score_time"].mean()),
            }
        }

    def _aggregate_metrics(self, results: dict) -> dict:
        metrics = {}
        for metric in self.config.metrics:
            scores = results[f"test_{metric}"]
            metrics[metric] = {
                "mean": float(scores.mean()),
                "std": float(scores.std()),
                "folds": scores.tolist(),
            }
            
        return metrics