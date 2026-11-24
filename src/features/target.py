import logging

import pandas as pd

from src.data.config import TargetConfig
from src.features.base import BaseTargetBuilder


logger = logging.getLogger(__name__)

class RepurchaseTargetBuilder(BaseTargetBuilder):
    def __init__(self, config: TargetConfig):
        self.config = config

    def build(self, sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
        logger.info("Target engineering started")

        completed = sources["completed_orders"]
        
        cutoff = pd.Timestamp(self.config.observation_end)
        horizon = pd.Timedelta(days=self.config.horizon_days)

        # Only observed transactions
        completed = completed.loc[completed["created_at"] <= cutoff]

        # One row per distinct order
        orders = (completed.groupby(["user_id", "order_id"], as_index=False)
                        .agg(purchase_date=("created_at", "min"))
                        .sort_values(["user_id", "purchase_date", "order_id"])
                    )

        orders["purchase_number"] = (orders.groupby("user_id").cumcount() + 1)

        first = orders.loc[orders["purchase_number"] == 1, ["user_id", "order_id", "purchase_date"]].rename(
            columns={
                "order_id": "first_order_id",
                "purchase_date": "first_purchase_date"
            }
        )

        second = orders.loc[orders["purchase_number"] == 2, ["user_id", "purchase_date"]].rename(
            columns={
                "purchase_date": "second_purchase_date"
            }
        )

        target = first.merge(second, on="user_id", how="left",
            validate="one_to_one"
        )

        # Remove right-censored observations
        eligible = (target["first_purchase_date"] + horizon <= cutoff)

        excluded = int((~eligible).sum())

        target = target.loc[eligible].copy()

        delta = (target["second_purchase_date"] - target["first_purchase_date"])

        target["days_to_repurchase"] = (delta.dt.total_seconds() / 86400)

        target[self.config.name] = (delta.notna() & delta.le(horizon)).astype("int8")

        if target.empty:
            raise ValueError("No eligible target observations")

        logger.info(
            "Target created | users=%s | positive_rate=%.4f "
            "| excluded_incomplete_horizon=%s",
            len(target),
            target[self.config.name].mean(),
            excluded,
        )

        return target