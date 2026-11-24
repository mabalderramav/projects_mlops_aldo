import logging
from pathlib import Path

import pandas as pd

from src.data.config import SourcesConfig


logger = logging.getLogger(__name__)

def load_data(path: Path) -> pd.DataFrame:
    """Load a prepared training dataset."""
    
    logger.info("Loading dataset: %s", path)
    
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )
    
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, low_memory=False)
    elif path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}"
                         )
    if df.empty:
        raise ValueError(f"The dataset is empty {path}")

    logger.info(
        "Dataset loaded | rows=%s | columns=%s",
        df.shape[0],
        df.shape[1],
    )
    return df

def load_sources(config: SourcesConfig) -> dict[str, pd.DataFrame]:
    return {
        name: load_data(path)
        for name, path in config.model_dump().items()
    }

def split_features_target(df: pd.DataFrame,target: str,) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features from the target column."""

    if target not in df.columns:
        raise ValueError(
            f"Target column not found: {target}"
        )

    X = df.drop(columns=[target])
    y = df[target]

    return X, y