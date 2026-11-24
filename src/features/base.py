from abc import ABC, abstractmethod

import pandas as pd


class BaseTargetBuilder(ABC):
    @abstractmethod
    def build(self, sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
        pass


class BaseFeatureBuilder(ABC):
    @abstractmethod
    def build(self, sources: dict[str, pd.DataFrame], anchors: pd.DataFrame) -> pd.DataFrame:
        pass