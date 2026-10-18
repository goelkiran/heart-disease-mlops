"""
Data Preparation and Feature Engineering Pipeline for Heart Disease UCI Dataset
AIML ZG523 - MLOps Assignment 1
"""

from pathlib import Path
from typing import Dict, Any, Tuple, List
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

RAW_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "heart.csv"
PROCESSED_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "heart_cleaned.csv"
PREPROCESSOR_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "preprocessor.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Feature column definitions
NUMERICAL_COLS: List[str] = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLS: List[str] = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURE_COLS: List[str] = NUMERICAL_COLS + CATEGORICAL_COLS
TARGET_COL: str = "target"


def load_raw_dataset(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw dataset from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at {path}. Run download_data.py first.")
    df = pd.read_csv(path)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataset:
    - Standardize column names
    - Convert multi-class angiographic status (0, 1, 2, 3, 4) into binary target (0: Absence, 1: Presence)
    - Enforce proper numeric data types
    """
    df = df.copy()

    # Column name mapping
    if "num" in df.columns and "target" not in df.columns:
        df.rename(columns={"num": "target"}, inplace=True)

    # Convert target to binary classification: 0 = No heart disease, 1 = Heart disease present
    df[TARGET_COL] = (df[TARGET_COL] > 0).astype(int)

    # Coerce features to numeric
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Construct a leak-free ColumnTransformer pipeline:
    - Numerical: Median imputation + Standard scaling
    - Categorical: Most frequent imputation + One-Hot Encoding (handle_unknown='ignore')
    """
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_COLS),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
        ],
        remainder="drop",
    )

    return preprocessor


def prepare_data(
    raw_path: Path = RAW_DATA_PATH,
    processed_path: Path = PROCESSED_DATA_PATH,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """
    End-to-end data preparation workflow:
    1. Loads raw dataset
    2. Cleans & normalizes binary target
    3. Saves cleaned dataframe to data/processed/
    4. Performs stratified train/test split
    5. Returns split subsets and initialized preprocessor
    """
    raw_df = load_raw_dataset(raw_path)
    cleaned_df = clean_dataset(raw_df)

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(processed_path, index=False)
    print(f"[+] Saved cleaned dataset to {processed_path} (Shape: {cleaned_df.shape})")

    X = cleaned_df[FEATURE_COLS]
    y = cleaned_df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    preprocessor = build_preprocessor()

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "cleaned_df": cleaned_df,
    }


if __name__ == "__main__":
    data_dict = prepare_data()
    print("[+] Train set size:", data_dict["X_train"].shape)
    print("[+] Test set size:", data_dict["X_test"].shape)
    print("[+] Target class distribution in train set:\n", data_dict["y_train"].value_counts(normalize=True))
