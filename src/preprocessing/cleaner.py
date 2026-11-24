import logging

import pandas as pd

from src.data.config import PreprocessingConfig


logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self, config: PreprocessingConfig):
        self.config = config

    def clean(self, sources: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        logger.info("Data cleaning started")

        cleaned = {
            name: df.copy()
            for name, df in sources.items()
        }

        # Convert timestamps to UTC
        for name in ("orders", "order_details", "users"):
            cleaned[name]["created_at"] = pd.to_datetime(
                cleaned[name]["created_at"],
                format="mixed",
                utc=True,
                errors="raise",
            )

        details = cleaned["order_details"]

        # Validate numeric values
        details["sale_price"] = pd.to_numeric(details["sale_price"], errors="raise")

        if (details["sale_price"] < 0).any():
            raise ValueError(
                "Negative sale prices detected"
            )

        # Business rule: completed purchases
        completed = details.loc[details["status"] == self.config.order_status].copy()

        if completed.empty:
            raise ValueError("No completed purchases")

        cleaned["completed_orders"] = completed
        logger.info("Data cleaning completed | completed_items=%s", len(completed))
        return cleaned