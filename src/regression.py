"""Linear Regression House Price Prediction Utilities.

Provides reusable data loading, preprocessing pipeline creation,
model evaluation, and diagnostic routines.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERICAL_FEATURES: List[str] = [
    "area",
    "bedrooms",
    "bathrooms",
    "stories",
    "parking",
]

CATEGORICAL_FEATURES: List[str] = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
    "furnishingstatus",
]

TARGET_COLUMN: str = "price"


def load_dataset(filepath: str = "dataset/Housing.csv") -> pd.DataFrame:
    """Load the housing dataset from relative CSV path."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
    return pd.read_csv(path)


def split_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataset into train and test feature matrices and targets."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def build_preprocessor() -> ColumnTransformer:
    """Construct ColumnTransformer for numerical passthrough and one-hot encoding."""
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERICAL_FEATURES),
            (
                "cat",
                OneHotEncoder(
                    drop="first", sparse_output=False, handle_unknown="ignore"
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_multiple_regression_pipeline() -> Pipeline:
    """Build full scikit-learn pipeline for Multiple Linear Regression."""
    preprocessor = build_preprocessor()
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model"
) -> Dict[str, float]:
    """Compute MAE, MSE, RMSE, and R2 score."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
    }


def save_model_artifact(artifact, filepath: str) -> Path:
    """Persist trained model or pipeline to disk using joblib."""
    dest = Path(filepath)
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, dest)
    return dest


def load_model_artifact(filepath: str):
    """Load serialized model or pipeline artifact."""
    dest = Path(filepath)
    if not dest.exists():
        raise FileNotFoundError(f"Artifact not found: {filepath}")
    return joblib.load(dest)

