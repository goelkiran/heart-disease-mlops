"""
Unit Tests for Data Ingestion, Integrity, and Preprocessing
AIML ZG523 - MLOps Assignment 1
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure src/ is importable
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_prep import clean_dataset, build_preprocessor, FEATURE_COLS, NUMERICAL_COLS, CATEGORICAL_COLS


@pytest.fixture
def sample_raw_dataframe():
    """Create a synthetic mini-batch reflecting raw Cleveland dataset schema."""
    data = {
        "age": [63, 67, 37, 41],
        "sex": [1, 1, 1, 0],
        "cp": [1, 4, 3, 2],
        "trestbps": [145, 160, 130, 130],
        "chol": [233, 286, 250, 204],
        "fbs": [1, 0, 0, 0],
        "restecg": [2, 2, 0, 2],
        "thalach": [150, 108, 187, 172],
        "exang": [0, 1, 0, 0],
        "oldpeak": [2.3, 1.5, 3.5, 1.4],
        "slope": [3, 2, 3, 1],
        "ca": [0.0, 3.0, 0.0, np.nan],  # test missing value in ca
        "thal": [6.0, 3.0, np.nan, 3.0], # test missing value in thal
        "target": [0, 2, 0, 1],         # multi-class raw targets
    }
    return pd.DataFrame(data)


def test_clean_dataset_binary_target(sample_raw_dataframe):
    """Verify that multi-class targets (0, 1, 2, 3) are converted to binary {0, 1}."""
    df_clean = clean_dataset(sample_raw_dataframe)
    unique_targets = set(df_clean["target"].unique())
    assert unique_targets.issubset({0, 1}), f"Targets must be binary {0, 1}, got {unique_targets}"
    assert df_clean["target"].iloc[0] == 0  # 0 stays 0
    assert df_clean["target"].iloc[1] == 1  # 2 becomes 1
    assert df_clean["target"].iloc[3] == 1  # 1 stays 1


def test_feature_columns_presence(sample_raw_dataframe):
    """Ensure all required numerical and categorical features exist."""
    df_clean = clean_dataset(sample_raw_dataframe)
    for col in FEATURE_COLS:
        assert col in df_clean.columns, f"Required column {col} missing from cleaned dataset."


def test_preprocessor_imputation_and_scaling(sample_raw_dataframe):
    """Verify that the ColumnTransformer handles NaN imputation and outputs numeric arrays."""
    df_clean = clean_dataset(sample_raw_dataframe)
    X = df_clean[FEATURE_COLS]

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)

    # Must contain zero NaN or inf values after transformation
    assert not np.isnan(transformed).any(), "Transformed array contains NaN values."
    assert not np.isinf(transformed).any(), "Transformed array contains infinite values."
    assert transformed.shape[0] == len(sample_raw_dataframe)
    assert transformed.shape[1] >= len(FEATURE_COLS)
