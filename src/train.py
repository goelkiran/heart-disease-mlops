"""
Model Training and Experiment Tracking Pipeline using MLflow
Trains Logistic Regression & Random Forest classifiers with Hyperparameter Tuning,
Stratified K-Fold Cross-Validation, and Comprehensive Artifact Logging.
AIML ZG523 - MLOps Assignment 1
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline

import mlflow
import mlflow.sklearn

from data_prep import prepare_data, RANDOM_STATE, FEATURE_COLS

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "report" / "figures"
MLFLOW_TRACKING_DIR = Path(__file__).resolve().parent.parent / "mlruns"


def setup_mlflow():
    """Initialize local SQLite-backed MLflow tracking server and experiment."""
    db_path = (Path(__file__).resolve().parent.parent / "mlflow.db").as_posix()
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("heart-disease-uci-experiment")
    print(f"[*] MLflow tracking URI set to: {mlflow.get_tracking_uri()}")


def train_logistic_regression(
    preprocessor, X_train, y_train, X_test, y_test
) -> Tuple[Pipeline, Dict[str, float]]:
    """
    Train and tune Logistic Regression classifier with L2 regularization.
    Logs run parameters, metrics, confusion matrix, and ROC curve to MLflow.
    """
    print("\n" + "=" * 60)
    print("[*] Training Model 1: Regularized Logistic Regression...")
    print("=" * 60)

    with mlflow.start_run(run_name="Logistic_Regression_L2"):
        base_lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        pipeline = Pipeline([("preprocess", preprocessor), ("classifier", base_lr)])

        param_grid = {
            "classifier__C": [0.01, 0.1, 1.0, 5.0, 10.0],
            "classifier__solver": ["liblinear", "lbfgs"],
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        best_pipe = grid_search.best_estimator_
        best_params = grid_search.best_params_
        cv_score = grid_search.best_score_

        print(f"[+] Best CV ROC-AUC: {cv_score:.4f} with params: {best_params}")
        mlflow.log_params(best_params)
        mlflow.log_metric("cv_roc_auc_mean", cv_score)

        # Evaluate on held-out test set
        y_pred = best_pipe.predict(X_test)
        y_proba = best_pipe.predict_proba(X_test)[:, 1]

        test_metrics = {
            "test_accuracy": float(accuracy_score(y_test, y_pred)),
            "test_precision": float(precision_score(y_test, y_pred)),
            "test_recall": float(recall_score(y_test, y_pred)),
            "test_f1": float(f1_score(y_test, y_pred)),
            "test_roc_auc": float(roc_auc_score(y_test, y_proba)),
        }

        for metric_name, val in test_metrics.items():
            mlflow.log_metric(metric_name, val)
            print(f"    {metric_name}: {val:.4f}")

        # Confusion Matrix artifact
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(cm, display_labels=["Absence", "Presence"]).plot(
            ax=ax, cmap="Blues"
        )
        ax.set_title("Logistic Regression Confusion Matrix")
        cm_path = FIGURES_DIR / "lr_confusion_matrix.png"
        plt.tight_layout()
        plt.savefig(cm_path, dpi=300)
        plt.close()
        mlflow.log_artifact(str(cm_path), artifact_path="evaluation_plots")

        # ROC Curve artifact
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig_roc, ax = plt.subplots(figsize=(6, 5))
        ax.plot(
            fpr,
            tpr,
            label=f"Logistic Regression (AUC = {test_metrics['test_roc_auc']:.3f})",
            color="#2b5c8f",
            lw=2,
        )
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve - Logistic Regression")
        ax.legend(loc="lower right")
        roc_path = FIGURES_DIR / "lr_roc_curve.png"
        plt.tight_layout()
        plt.savefig(roc_path, dpi=300)
        plt.close()
        mlflow.log_artifact(str(roc_path), artifact_path="evaluation_plots")

        # Log Scikit-Learn Model Artifact to MLflow
        mlflow.sklearn.log_model(best_pipe, artifact_path="model", serialization_format="cloudpickle")

    return best_pipe, test_metrics


def train_random_forest(
    preprocessor, X_train, y_train, X_test, y_test
) -> Tuple[Pipeline, Dict[str, float]]:
    """
    Train and tune Random Forest classifier with hyperparameter search.
    Logs run parameters, metrics, confusion matrix, and ROC curve to MLflow.
    """
    print("\n" + "=" * 60)
    print("[*] Training Model 2: Tuned Random Forest Classifier...")
    print("=" * 60)

    with mlflow.start_run(run_name="Random_Forest_Tuned"):
        base_rf = RandomForestClassifier(random_state=RANDOM_STATE)
        pipeline = Pipeline([("preprocess", preprocessor), ("classifier", base_rf)])

        param_grid = {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__max_depth": [4, 6, 8, None],
            "classifier__min_samples_split": [2, 4, 8],
            "classifier__class_weight": ["balanced", None],
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        best_pipe = grid_search.best_estimator_
        best_params = grid_search.best_params_
        cv_score = grid_search.best_score_

        print(f"[+] Best CV ROC-AUC: {cv_score:.4f} with params: {best_params}")
        mlflow.log_params(best_params)
        mlflow.log_metric("cv_roc_auc_mean", cv_score)

        # Evaluate on held-out test set
        y_pred = best_pipe.predict(X_test)
        y_proba = best_pipe.predict_proba(X_test)[:, 1]

        test_metrics = {
            "test_accuracy": float(accuracy_score(y_test, y_pred)),
            "test_precision": float(precision_score(y_test, y_pred)),
            "test_recall": float(recall_score(y_test, y_pred)),
            "test_f1": float(f1_score(y_test, y_pred)),
            "test_roc_auc": float(roc_auc_score(y_test, y_proba)),
        }

        for metric_name, val in test_metrics.items():
            mlflow.log_metric(metric_name, val)
            print(f"    {metric_name}: {val:.4f}")

        # Confusion Matrix artifact
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(cm, display_labels=["Absence", "Presence"]).plot(
            ax=ax, cmap="Greens"
        )
        ax.set_title("Random Forest Confusion Matrix")
        cm_path = FIGURES_DIR / "rf_confusion_matrix.png"
        plt.tight_layout()
        plt.savefig(cm_path, dpi=300)
        plt.close()
        mlflow.log_artifact(str(cm_path), artifact_path="evaluation_plots")

        # ROC Curve artifact
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig_roc, ax = plt.subplots(figsize=(6, 5))
        ax.plot(
            fpr,
            tpr,
            label=f"Random Forest (AUC = {test_metrics['test_roc_auc']:.3f})",
            color="#2ca02c",
            lw=2,
        )
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve - Random Forest")
        ax.legend(loc="lower right")
        roc_path = FIGURES_DIR / "rf_roc_curve.png"
        plt.tight_layout()
        plt.savefig(roc_path, dpi=300)
        plt.close()
        mlflow.log_artifact(str(roc_path), artifact_path="evaluation_plots")

        # Log Scikit-Learn Model Artifact to MLflow
        mlflow.sklearn.log_model(best_pipe, artifact_path="model", serialization_format="cloudpickle")

    return best_pipe, test_metrics


def generate_comparative_roc_curve(lr_pipe, rf_pipe, X_test, y_test):
    """Generate joint ROC curve comparing both models for the report."""
    y_proba_lr = lr_pipe.predict_proba(X_test)[:, 1]
    y_proba_rf = rf_pipe.predict_proba(X_test)[:, 1]

    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)

    auc_lr = roc_auc_score(y_test, y_proba_lr)
    auc_rf = roc_auc_score(y_test, y_proba_rf)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {auc_lr:.3f})", color="#2b5c8f", lw=2)
    plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.3f})", color="#2ca02c", lw=2)
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.7)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontweight="bold")
    plt.title("Model Comparison: ROC Curves on Held-Out Test Set", pad=15, fontweight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    out_path = FIGURES_DIR / "model_comparison_roc.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Saved comparative ROC plot to {out_path}")


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    setup_mlflow()

    # Load and prepare data
    data = prepare_data()
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    preprocessor = data["preprocessor"]

    # Train Model 1 (Logistic Regression)
    lr_pipe, lr_metrics = train_logistic_regression(preprocessor, X_train, y_train, X_test, y_test)

    # Train Model 2 (Random Forest)
    rf_pipe, rf_metrics = train_random_forest(preprocessor, X_train, y_train, X_test, y_test)

    # Comparative evaluation
    generate_comparative_roc_curve(lr_pipe, rf_pipe, X_test, y_test)

    # Select champion model based on ROC-AUC / F1
    if rf_metrics["test_roc_auc"] >= lr_metrics["test_roc_auc"]:
        champion_name = "RandomForestClassifier"
        champion_pipe = rf_pipe
        champion_metrics = rf_metrics
    else:
        champion_name = "LogisticRegression"
        champion_pipe = lr_pipe
        champion_metrics = lr_metrics

    print("\n" + "=" * 60)
    print(f"[*] Champion Model Selected: {champion_name}")
    print(f"    Test Accuracy: {champion_metrics['test_accuracy']:.4f}")
    print(f"    Test Recall:   {champion_metrics['test_recall']:.4f}")
    print(f"    Test ROC-AUC:  {champion_metrics['test_roc_auc']:.4f}")
    print("=" * 60)

    # Save champion model artifact for Docker serving and inference
    model_save_path = ARTIFACTS_DIR / "best_model.joblib"
    joblib.dump(champion_pipe, model_save_path)
    print(f"[+] Saved champion pipeline artifact to: {model_save_path}")

    # Save metrics metadata JSON
    metadata = {
        "champion_model": champion_name,
        "feature_columns": FEATURE_COLS,
        "logistic_regression_metrics": lr_metrics,
        "random_forest_metrics": rf_metrics,
        "champion_metrics": champion_metrics,
    }
    meta_path = ARTIFACTS_DIR / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"[+] Saved model metadata to: {meta_path}")


if __name__ == "__main__":
    main()
