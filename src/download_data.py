"""
Data Acquisition Script for Heart Disease UCI Dataset
AIML ZG523 - MLOps Assignment 1
"""

import os
import sys
import urllib.request
from pathlib import Path
import pandas as pd

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_FILE = RAW_DATA_DIR / "heart.csv"

# Primary and fallback URLs for the Heart Disease dataset (Cleveland database)
UCI_PRIMARY_URL = "https://archive.ics.uci.edu/static/public/45/data.csv"
UCI_PROCESSED_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# Standard 14 attribute names defined in the UCI Heart Disease documentation
COLUMN_NAMES = [
    "age",      # age in years
    "sex",      # sex (1 = male; 0 = female)
    "cp",       # chest pain type (1, 2, 3, 4)
    "trestbps", # resting blood pressure (in mm Hg on admission to the hospital)
    "chol",     # serum cholestoral in mg/dl
    "fbs",      # fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
    "restecg",  # resting electrocardiographic results (0, 1, 2)
    "thalach",  # maximum heart rate achieved
    "exang",    # exercise induced angina (1 = yes; 0 = no)
    "oldpeak",  # ST depression induced by exercise relative to rest
    "slope",    # the slope of the peak exercise ST segment
    "ca",       # number of major vessels (0-3) colored by flourosopy
    "thal",     # 3 = normal; 6 = fixed defect; 7 = reversable defect
    "target"    # diagnosis of heart disease (angiographic disease status)
]


def download_dataset(output_path: Path = DATA_FILE) -> Path:
    """
    Download the UCI Heart Disease dataset and save it to the raw data directory.
    Attempts primary modern UCI download first; falls back to processed.cleveland.data.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[*] Checking/fetching dataset at {output_path}...")
    try:
        print(f"[*] Attempting download from primary URL: {UCI_PRIMARY_URL}")
        urllib.request.urlretrieve(UCI_PRIMARY_URL, output_path)
        df = pd.read_csv(output_path)
        # Check if column names match or need renaming
        if "num" in df.columns and "target" not in df.columns:
            df.rename(columns={"num": "target"}, inplace=True)
            df.to_csv(output_path, index=False)
        print(f"[+] Successfully downloaded primary UCI dataset ({len(df)} rows, {len(df.columns)} columns).")
    except Exception as e:
        print(f"[!] Primary URL failed ({e}). Falling back to processed.cleveland.data...")
        try:
            urllib.request.urlretrieve(UCI_PROCESSED_URL, output_path)
            # Legacy format has no headers and uses '?' for missing values
            df = pd.read_csv(output_path, names=COLUMN_NAMES, na_values="?")
            df.to_csv(output_path, index=False)
            print(f"[+] Successfully downloaded and formatted fallback dataset ({len(df)} rows).")
        except Exception as e2:
            print(f"[-] Failed to download from fallback URL: {e2}")
            sys.exit(1)

    print(f"[+] Dataset saved to: {output_path}")
    print("\n[+] Dataset Overview:")
    print(df.info())
    print("\n[+] First 5 records:")
    print(df.head())
    return output_path


if __name__ == "__main__":
    download_dataset()
