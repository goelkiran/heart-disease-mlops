"""
Exploratory Data Analysis (EDA) Script
Generates publication-quality visualizations for the Heart Disease UCI dataset.
AIML ZG523 - MLOps Assignment 1
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Path resolution & Single Source of Truth import from data_prep
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from data_prep import NUMERICAL_COLS
except ImportError:
    from src.data_prep import NUMERICAL_COLS

FIGURES_DIR = Path(__file__).resolve().parent.parent / "report" / "figures"
PROCESSED_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "heart_cleaned.csv"

# ==============================================================================
# Semantic Design Tokens / Color Palette
# ==============================================================================
COLOR_ABSENCE = "#2b5c8f"        # Deep Steel Blue: Absence of disease (Class 0)
COLOR_PRESENCE = "#d95f02"       # Burnt Orange: Presence of disease (Class 1)
COLOR_ABSENCE_LIGHT = "#7ea6c9"  # Muted Blue: Boxplot fill
COLOR_PRESENCE_LIGHT = "#e59866" # Muted Orange: Boxplot fill
CLASS_COLORS = [COLOR_ABSENCE, COLOR_PRESENCE]


def setup_plotting_style():
    """Configure clean matplotlib aesthetic parameters."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["figure.dpi"] = 300


def generate_eda_figures():
    """Generate all mandatory EDA figures for the technical report."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    setup_plotting_style()

    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"[*] Loaded dataset with shape {df.shape} for EDA.")

    # 1. Target Class Balance
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df["target"].value_counts().sort_index()
    labels = ["Absence (0)", "Presence (1)"]
    bars = ax.bar(labels, counts.values, color=CLASS_COLORS, width=0.55, edgecolor="black", linewidth=1.2)

    for bar in bars:
        yval = bar.get_height()
        pct = (yval / len(df)) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 4,
            f"{yval} ({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    ax.set_title("Heart Disease Target Distribution (Class Balance)", pad=15, fontweight="bold")
    ax.set_ylabel("Patient Count", fontweight="bold")
    ax.set_ylim(0, max(counts.values) + 30)
    plt.tight_layout()
    fig1_path = FIGURES_DIR / "class_balance.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"[+] Saved {fig1_path}")

    # 2. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = df.corr()
    cax = ax.matshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax, shrink=0.8)

    ticks = np.arange(len(corr.columns))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(corr.columns, rotation=45, ha="left", fontsize=10)
    ax.set_yticklabels(corr.columns, fontsize=10)

    # Annotate correlation values
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            val = corr.iloc[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

    ax.set_title("Feature Correlation Matrix (Pearson r)", pad=30, fontweight="bold")
    plt.tight_layout()
    fig2_path = FIGURES_DIR / "correlation_heatmap.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"[+] Saved {fig2_path}")

    # 3. Numerical Feature Histograms & Distributions by Class (Single Source of Truth)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for idx, feature in enumerate(NUMERICAL_COLS):
        ax = axes[idx]
        df[df["target"] == 0][feature].hist(
            ax=ax, bins=15, alpha=0.6, color=COLOR_ABSENCE, label="Absence (0)", edgecolor="black"
        )
        df[df["target"] == 1][feature].hist(
            ax=ax, bins=15, alpha=0.6, color=COLOR_PRESENCE, label="Presence (1)", edgecolor="black"
        )
        ax.set_title(f"Distribution of {feature.upper()}", fontweight="bold", fontsize=11)
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.legend(loc="upper right")

    # Hide unused 6th subplot
    axes[5].axis("off")
    plt.suptitle("Numerical Feature Distributions Stratified by Heart Disease Status", fontsize=15, fontweight="bold")
    plt.tight_layout()
    fig3_path = FIGURES_DIR / "numerical_distributions.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"[+] Saved {fig3_path}")

    # 4. Boxplots of Key Physiological Indicators
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    key_indicators = [
        ("thalach", "Max Heart Rate Achieved (thalach)", "bpm"),
        ("oldpeak", "ST Depression (oldpeak)", "mm"),
        ("chol", "Serum Cholesterol (chol)", "mg/dl"),
    ]

    for idx, (col, title, unit) in enumerate(key_indicators):
        ax = axes[idx]
        data_to_plot = [df[df["target"] == 0][col].dropna(), df[df["target"] == 1][col].dropna()]
        bplot = ax.boxplot(
            data_to_plot,
            patch_artist=True,
            tick_labels=["Absence (0)", "Presence (1)"],
            medianprops=dict(color="black", linewidth=1.5),
        )
        bplot["boxes"][0].set_facecolor(COLOR_ABSENCE_LIGHT)
        bplot["boxes"][1].set_facecolor(COLOR_PRESENCE_LIGHT)
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_ylabel(unit, fontsize=10)

    plt.suptitle("Key Physiological Differences Between Disease Absence and Presence", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig4_path = FIGURES_DIR / "boxplots_by_target.png"
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"[+] Saved {fig4_path}")

    print("\n[+] All EDA figures generated successfully.")


if __name__ == "__main__":
    generate_eda_figures()
