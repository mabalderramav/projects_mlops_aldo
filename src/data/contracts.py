import logging

from dataclasses import dataclass
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class DataContract(BaseModel):
    model_config = ConfigDict(frozen=True)
    required_columns: list[str] = Field(default_factory=list)
    primary_key: str
    non_nullable: list[str] = Field(default_factory=list)

RAW_CONTRACTS = {
    "orders": DataContract(
        required_columns=["order_id", "user_id", "status", "created_at"],
        primary_key="order_id",
        non_nullable=["order_id", "user_id", "created_at"]
    ),
    "order_details": DataContract(
        required_columns=["id", "order_id", "user_id", "product_id", "status", "created_at", "sale_price"],
        primary_key="id",
        non_nullable=["id", "order_id", "user_id", "product_id", "status", "created_at", "sale_price"]
    ),
    "products": DataContract(
        required_columns=["id", "cost", "retail_price", "category", "brand", "department"],
        primary_key="id",
        non_nullable=["id"],
    ),
    "users": DataContract(
        required_columns=["id", "age", "gender", "state", "country", "traffic_source", "created_at"],
        primary_key="id",
        non_nullable=["id", "created_at"]
    ),
}


class DataContractValidator:
    def validate(self, df: pd.DataFrame, contract: DataContract, name: str) -> None:
        missing = (set(contract.required_columns) - set(df.columns))

        if missing:
            raise ValueError(f"{name}: missing columns {sorted(missing)}")

        for column in contract.non_nullable:
            logger.debug("Validating column | dataset=%s | column=%s", name, column)
            if df[column].isna().any():
                raise ValueError(f"{name}: null values in {column}")

        if df[contract.primary_key].duplicated().any():
            raise ValueError(
                f"{name}: duplicated primary key "
                f"{contract.primary_key}"
            )

        logger.info("Data contract validated: %s", name)

    def validate_sources(self, sources: dict[str, pd.DataFrame]) -> None:

        for name, contract in RAW_CONTRACTS.items():
            self.validate(df=sources[name], contract=contract, name=name)

        # Referential integrity
        checks = [
            ("order_details", "order_id", "orders", "order_id"),
            ("order_details", "product_id", "products", "id"),
            ("order_details", "user_id", "users", "id"),
            ("orders", "user_id", "users", "id"),
        ]

        for child, key, parent, parent_key in checks:
            invalid = ~sources[child][key].isin(sources[parent][parent_key])

            if invalid.any():
                raise ValueError(f"Referential integrity failed: {child}.{key} -> {parent}.{parent_key}")
            
            logger.debug("Referential integrity validated: %s.%s -> %s.%s", child, key, parent, parent_key)
        logger.info("All source data contracts validated")