import logging

import pandas as pd

from sklearn.model_selection import train_test_split

from src.data.config import SplitConfig


logger = logging.getLogger(__name__)

class DatasetSplitter:
    def __init__(self, config: SplitConfig):
        self.config = config

    def split(self, df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("Dataset splitting started")

        if set(df[target].unique()) != {0, 1}:
            raise ValueError("Both target classes must be present")

        stratify = (
            df[target]
            if self.config.stratify
            else None
        )

        train, test = train_test_split(
            df,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=stratify,
        )

        if not train.index.intersection(test.index).empty:
            raise ValueError("Train and test observations overlap")

        for name, partition in (("train", train), ("test", test)):
            if set(partition[target].unique()) != {0, 1}:
                raise ValueError(f"Both classes must be present in {name}")

        logger.info("Dataset split completed | train=%s | test=%s", len(train), len(test))
        logger.info("Positive rates | train=%.4f | test=%.4f", train[target].mean(), test[target].mean())

        return (
            train.reset_index(drop=True),
            test.reset_index(drop=True),
        )