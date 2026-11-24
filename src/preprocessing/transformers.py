import pandas as pd

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder


def low_cardinality_selector(X: pd.DataFrame) -> list[str]:
    categorical = X.select_dtypes(include=["object", "category", "string"]).columns

    return [
        column
        for column in categorical
        if X[column].nunique() <= 5
    ]

def build_lr_preprocessor() -> ColumnTransformer:
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", numerical_pipeline, make_column_selector(dtype_include="number")),
        ("cat", categorical_pipeline, low_cardinality_selector)
    ], remainder="drop")

def build_tree_preprocessor() -> ColumnTransformer:
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        ),
    ])

    return ColumnTransformer([
        ("num", numerical_pipeline, make_column_selector(dtype_include="number")),
        ("cat", categorical_pipeline, make_column_selector(dtype_include=["object", "category", "string"]))
    ], remainder="drop")