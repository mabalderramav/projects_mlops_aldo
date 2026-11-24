import logging

import pandas as pd

from src.features.base import BaseFeatureBuilder


logger = logging.getLogger(__name__)

def safe_mode(series: pd.Series):
    modes = series.mode()
    return modes.iloc[0] if not modes.empty else "Unknown"

class PurchaseFeatureBuilder(BaseFeatureBuilder):
    def build(self, sources: dict[str, pd.DataFrame], anchors: pd.DataFrame) -> pd.DataFrame:

        logger.info("Feature engineering started")

        completed = sources["completed_orders"]
        products = sources["products"]
        users = sources["users"]

        # Only use information identifying the first purchase
        first_orders = anchors[["user_id", "first_order_id", "first_purchase_date"]].copy()

        # 1. First order details
        items = first_orders.merge(
            completed,
            left_on=["user_id", "first_order_id"],
            right_on=["user_id", "order_id"],
            how="inner",
            validate="one_to_many"
        )

        # 2. Monetary features
        monetary = (
            items.groupby("user_id", as_index=False)
            .agg(
                first_order_items=("product_id", "size"),
                first_order_total_value=("sale_price", "sum"),
                first_order_avg_price=("sale_price", "mean"),
                first_order_max_price=("sale_price", "max"),
                first_order_min_price=("sale_price", "min")
            )
        )

        # 3. Product information
        unique_products = items[["user_id", "product_id"]].drop_duplicates()

        catalog = products[["id", "cost", "retail_price", "category", "brand", "department"]]

        product_details = unique_products.merge(
            catalog,
            left_on="product_id",
            right_on="id",
            how="left",
            validate="many_to_one",
        )

        # 4. Product features
        product_features = (
            product_details.groupby("user_id", as_index=False)
            .agg(
                first_order_avg_cost=("cost", "mean"),
                first_order_avg_retail_price=("retail_price", "mean"),
                first_order_categories=("category", "nunique"),
                first_order_brands=("brand", "nunique"),
                first_order_departments=("department", "nunique"),
                most_common_category=("category", safe_mode),
                most_common_brand=("brand", safe_mode),
                most_common_department=("department", safe_mode)
            )
        )

        # 5. User information
        user_features = users[["id", "age", "gender", "state", "country", "traffic_source", "created_at"]].rename(
            columns={
                "id": "user_id",
                "created_at": "registration_date"
            }
        )

        # 6. Join features
        features = (
            first_orders.merge(
                monetary,
                on="user_id",
                how="left",
                validate="one_to_one",
            ).merge(
                product_features,
                on="user_id",
                how="left",
                validate="one_to_one",
            ).merge(
                user_features,
                on="user_id",
                how="left",
                validate="one_to_one",
            )
        )

        # 7. Temporal features
        purchase_date = features["first_purchase_date"]
        features["first_purchase_hour"] = (purchase_date.dt.hour)
        features["first_purchase_day_of_week"] = (purchase_date.dt.dayofweek)
        features["first_purchase_month"] = (purchase_date.dt.month)
        features["first_purchase_year"] = (purchase_date.dt.year)
        features["days_registration_to_first_purchase"] = (purchase_date - features["registration_date"]).dt.days

        if features["user_id"].duplicated().any():
            raise ValueError("Duplicated user features")

        if features["first_order_items"].isna().any():
            raise ValueError("Missing first order details")

        logger.info(
            "Feature engineering completed | "
            "rows=%s | columns=%s",
            features.shape[0],
            features.shape[1],
        )

        return features