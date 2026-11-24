import logging
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

from src.models.base import BaseModel


logger = logging.getLogger(__name__)

class Evaluator:
    """Evaluate binary classification models."""

    def __init__(self, model: BaseModel):
        self.model = model

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:

        logger.info(
            "Evaluation started | model=%s",
            type(self.model).__name__,
        )

        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, y_proba)),
            "average_precision": float(average_precision_score(y, y_proba)),
        }

        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        metrics["confusion_matrix"] = {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }

        logger.info(
            "Evaluation completed | roc_auc=%.4f | f1=%.4f",
            metrics["roc_auc"],
            metrics["f1_score"],
        )

        return metrics