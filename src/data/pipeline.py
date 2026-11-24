import json
import logging

from pathlib import Path

from src.data.config import load_config
from src.data.loader import load_sources
from src.data.contracts import DataContractValidator
from src.data.splitter import DatasetSplitter
from src.preprocessing.cleaner import DataCleaner
from src.features.base import (BaseTargetBuilder, BaseFeatureBuilder)
from src.features.target import RepurchaseTargetBuilder
from src.features.builder import PurchaseFeatureBuilder

logger = logging.getLogger(__name__)

class DataPipeline:

    def __init__(self, config_path: Path, target_builder: BaseTargetBuilder | None = None, feature_builder: BaseFeatureBuilder | None = None):
        self.config = load_config(config_path)
        self.validator = DataContractValidator()
        self.cleaner = DataCleaner(self.config.preprocessing)
        self.target_builder = (target_builder if target_builder is not None else RepurchaseTargetBuilder(self.config.target))
        self.feature_builder = (feature_builder if feature_builder is not None else PurchaseFeatureBuilder())
        self.splitter = DatasetSplitter(self.config.split)

    def run(self) -> dict:
        logger.info("Data pipeline started")

        # 1. Data ingestion
        sources = load_sources(self.config.sources)

        # 2. Data validation
        self.validator.validate_sources(sources)

        # 3. Data cleaning
        sources = self.cleaner.clean(sources)

        # 4. Target engineering
        target_df = self.target_builder.build(sources)

        # 5. Feature engineering
        anchors = target_df[["user_id", "first_order_id", "first_purchase_date"]].copy()

        features_df = self.feature_builder.build(sources, anchors)

        # 6. Build modelling dataset
        target_name = self.config.target.name
        
        dataset = features_df.merge(
            target_df[["user_id", target_name]],
            on="user_id",
            how="inner",
            validate="one_to_one"
        )

        if len(dataset) != len(target_df):
            raise ValueError("Feature and target observations mismatch")

        # 7. Remove non-predictive columns
        dataset = dataset.drop(columns=["user_id", "first_order_id", "first_purchase_date", "registration_date"])

        # 8. Split dataset
        train, test = self.splitter.split(dataset, target_name)

        # 9. Persist datasets
        output = self.config.output
        output.train_path.parent.mkdir(parents=True, exist_ok=True)
        output.test_path.parent.mkdir(parents=True, exist_ok=True)

        train.to_parquet(output.train_path, index=False)
        test.to_parquet(output.test_path, index=False)

        # 10. Generate metadata
        metadata = {
            "target": target_name,
            "horizon_days": (self.config.target.horizon_days),
            "observation_end": (self.config.target.observation_end.isoformat()),
            "total_rows": len(dataset),
            "train_rows": len(train),
            "test_rows": len(test),
            "feature_count": len(dataset.columns) - 1,
            "positive_rate": {
                "train": float(train[target_name].mean()),
                "test": float(test[target_name].mean()),
            },
            "schema": {
                column: str(dtype)
                for column, dtype in dataset.dtypes.items()
            },
            "config": self.config.model_dump(mode="json")
        }

        output.metadata_path.parent.mkdir(parents=True, exist_ok=True)

        with output.metadata_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=4)

        logger.info("Training dataset saved: %s", output.train_path)
        logger.info("Testing dataset saved: %s", output.test_path)
        logger.info("Metadata saved: %s", output.metadata_path)
        logger.info("Data pipeline completed")

        return metadata