import json
from pathlib import Path

notebook_path = Path("Sem03/MLOps_AIMLZG523/EC1_Quizzes_Assignments/Assignment_01/notebooks/01_eda_and_experimentation.ipynb")

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🫀 MLOps Assignment 01: Exploratory Data Analysis & Model Development\n",
            "**Course**: AIML ZG523 (MLOps) — BITS Pilani WILP  \n",
            "**Student**: Kiran Kumar (`2025AE05377`)  \n",
            "**Dataset**: UCI Heart Disease (Cleveland Database)  \n",
            "\n",
            "This notebook walks through the data ingestion, statistical exploratory analysis, leak-free pipeline preprocessing, multi-model training (Logistic Regression vs. Random Forest), and MLflow experiment tracking."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 1,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import sys\n",
            "from pathlib import Path\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "import joblib\n",
            "\n",
            "# Add src to python path\n",
            "sys.path.append('../src')\n",
            "from data_prep import prepare_data, FEATURE_COLS, NUMERICAL_COLS, CATEGORICAL_COLS\n",
            "print('Environment initialized successfully.')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Data Ingestion & Schema Inspection"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 2,
        "metadata": {},
        "outputs": [],
        "source": [
            "data_path = Path('../data/processed/heart_cleaned.csv')\n",
            "df = pd.read_csv(data_path)\n",
            "print(f'Dataset Shape: {df.shape}')\n",
            "df.head()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 3,
        "metadata": {},
        "outputs": [],
        "source": [
            "df.info()\n",
            "df.describe().T"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Target Class Distribution (Class Balance)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 4,
        "metadata": {},
        "outputs": [],
        "source": [
            "plt.figure(figsize=(6, 4))\n",
            "counts = df['target'].value_counts()\n",
            "plt.bar(['Absence (0)', 'Presence (1)'], counts.values, color=['#2b5c8f', '#d95f02'], width=0.5)\n",
            "plt.title('Target Variable Distribution (Cleveland Clinic)')\n",
            "plt.ylabel('Patient Count')\n",
            "for i, v in enumerate(counts.values):\n",
            "    plt.text(i, v + 3, f'{v} ({v/len(df)*100:.1f}%)', ha='center', fontweight='bold')\n",
            "plt.ylim(0, max(counts.values) + 25)\n",
            "plt.grid(axis='y', linestyle='--', alpha=0.7)\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Correlation Heatmap Analysis"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 5,
        "metadata": {},
        "outputs": [],
        "source": [
            "plt.figure(figsize=(10, 8))\n",
            "corr = df.corr()\n",
            "plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)\n",
            "plt.colorbar(shrink=0.8)\n",
            "plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha='right')\n",
            "plt.yticks(range(len(corr.columns)), corr.columns)\n",
            "plt.title('Pearson Correlation Heatmap of Clinical Variables')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Key Physiological Risk Indicators (Distributions & Boxplots)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 6,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))\n",
            "key_vars = [('thalach', 'Max Heart Rate Achieved (bpm)'), ('oldpeak', 'ST Depression (mm)'), ('chol', 'Serum Cholesterol (mg/dl)')]\n",
            "for idx, (col, title) in enumerate(key_vars):\n",
            "    ax = axes[idx]\n",
            "    data_groups = [df[df['target']==0][col].dropna(), df[df['target']==1][col].dropna()]\n",
            "    ax.boxplot(data_groups, tick_labels=['Absence', 'Presence'], patch_artist=True)\n",
            "    ax.set_title(title, fontweight='bold')\n",
            "    ax.grid(True, linestyle='--', alpha=0.5)\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Model Inspection & Inference Verification\n",
            "Load champion serialized pipeline (`best_model.joblib`) and test inference."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 7,
        "metadata": {},
        "outputs": [],
        "source": [
            "model_path = Path('../artifacts/best_model.joblib')\n",
            "pipeline = joblib.load(model_path)\n",
            "print('Loaded Pipeline Steps:', [step[0] for step in pipeline.steps])\n",
            "\n",
            "sample_patient = pd.DataFrame([{\n",
            "    'age': 60, 'sex': 1, 'cp': 4, 'trestbps': 145, 'chol': 282, 'fbs': 0,\n",
            "    'restecg': 2, 'thalach': 142, 'exang': 1, 'oldpeak': 2.8, 'slope': 2, 'ca': 2.0, 'thal': 7.0\n",
            "}])\n",
            "\n",
            "pred = pipeline.predict(sample_patient)[0]\n",
            "proba = pipeline.predict_proba(sample_patient)[0][1]\n",
            "print(f'Prediction: {pred} (Heart Disease Present)' if pred == 1 else f'Prediction: {pred} (Heart Disease Absent)')\n",
            "print(f'Calculated Probability: {proba:.4f}')"
        ]
    }
]

nb_dict = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

notebook_path.parent.mkdir(parents=True, exist_ok=True)
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb_dict, f, indent=2)

print(f"Generated clean Jupyter notebook at: {notebook_path}")
