import logging
from pathlib import Path

import pandas as pd

from src.models.base import BaseModel


logger = logging.getLogger(__name__)

class Trainer:
    """Train models using a common abstract contract."""

    def __init__(self, model: BaseModel):
        self.model = model
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> BaseModel:
        logger.info("Training started | model=%s", type(self.model).__name__)
        self.model.fit(X, y)
        logger.info("Training completed")
        return self.model

    def save(self, path: Path) -> None:
        self.model.save(path)
        logger.info("Model artifact saved: %s", path)