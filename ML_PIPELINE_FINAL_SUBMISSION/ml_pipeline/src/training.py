"""
training.py
-----------
Multi-Model Training Module with SHAP Explainability for the ML Pipeline.

Responsibilities:
  - Prepare features and target variable
  - Train/test split with stratification
  - Train MULTIPLE models (Logistic Regression, Random Forest, XGBoost, LightGBM)
  - Evaluate all models with multiple metrics
  - Auto-select the best model based on ROC-AUC
  - Generate SHAP feature importance plots for explainability
  - Save versioned model artifacts with metadata
  - Track metrics across runs in a JSON registry

Design Decisions:
  - Multi-model arena: Train multiple models and let data decide the best one
  - SHAP values: Critical for healthcare - explains WHY a model makes predictions
  - Model versioning via timestamp ensures no model is ever overwritten
  - StandardScaler is fitted on train set ONLY (no data leakage)
"""

import json
import os
import pickle
import warnings
from datetime import datetime
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.utils.logger import get_logger, log_section_header, log_section_footer

warnings.filterwarnings('ignore', category=UserWarning)
logger = get_logger(__name__)

METRICS_REGISTRY_FILE = "models/metrics_registry.json"


def _load_metrics_registry() -> list:
    """Load the historical metrics registry from disk."""
    if os.path.exists(METRICS_REGISTRY_FILE):
        with open(METRICS_REGISTRY_FILE, "r") as f:
            return json.load(f)
    return []


def _save_metrics_registry(registry: list) -> None:
    """Save the updated metrics registry to disk."""
    os.makedirs(os.path.dirname(METRICS_REGISTRY_FILE), exist_ok=True)
    with open(METRICS_REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)


def _get_next_model_version(models_dir: str) -> Tuple[int, str]:
    """Determine the next model version number and build the versioned filename."""
    existing = [
        f for f in os.listdir(models_dir)
        if f.startswith("model_v") and f.endswith(".pkl")
    ]

    if not existing:
        version = 1
    else:
        versions = []
        for f in existing:
            try:
                v = int(f.split("_v")[1].split("_")[0])
                versions.append(v)
            except (IndexError, ValueError):
                pass
        version = max(versions) + 1 if versions else 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"model_v{version}_{timestamp}.pkl"
    return version, filename


def prepare_features(
    df: pd.DataFrame,
    target_column: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate features (X) from the target variable (y)."""
    drop_cols = [target_column]
    drop_cols += [c for c in df.columns if c.startswith("__")]

    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].copy()
    y = df[target_column].copy()
    y = y.astype(int)

    logger.info(f"  Feature columns ({len(feature_cols)}): {feature_cols}")
    logger.info(f"  Target column: '{target_column}'")
    logger.info(f"  Class distribution — 0: {(y==0).sum():,}, 1: {(y==1).sum():,}")

    return X, y


def train_single_model(
    model_name: str,
    model_params: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Any]:
    """
    Train and evaluate a single model.
    
    Returns:
        Dict with model object, predictions, probabilities, and metrics.
    """
    # Model mapping
    model_classes = {
        "logistic_regression": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "xgboost": XGBClassifier,
        "lightgbm": LGBMClassifier
    }
    
    if model_name not in model_classes:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Train
    model = model_classes[model_name](**model_params)
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Evaluate
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }
    
    return {
        "model": model,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "metrics": metrics
    }


def generate_shap_plots(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    model_name: str,
    output_dir: str = "models/explainability"
) -> str:
    """
    Generate SHAP feature importance plots for model explainability.
    
    Returns:
        Path to the saved SHAP plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Use TreeExplainer for tree-based models, LinearExplainer for linear models
        if model_name in ["random_forest", "xgboost", "lightgbm"]:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, X_train)
        
        shap_values = explainer.shap_values(X_test)
        
        # For binary classification, shap_values might be a list
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class
        
        # Create summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, show=False, plot_type="bar")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(output_dir, f"shap_{model_name}_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  SHAP plot saved → {plot_path}")
        return plot_path
        
    except Exception as e:
        logger.warning(f"  Could not generate SHAP plot for {model_name}: {e}")
        return None


def train_model(
    df: pd.DataFrame,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Main training entry point. Trains multiple models, selects best, and generates explainability.
    
    Pipeline:
      1. Prepare features and target
      2. Stratified train/test split
      3. Feature scaling (StandardScaler fit on train only)
      4. Train ALL configured models
      5. Compare models and select best by ROC-AUC
      6. Generate SHAP explainability plots
      7. Save best model + scaler artifact
      8. Update metrics registry
    """
    log_section_header(logger, "Stage 5: Multi-Model Training & Evaluation")

    training_cfg = config.get("training", {})
    paths_cfg = config.get("paths", {})

    target_col = training_cfg.get("target_column", "Outcome")
    test_size = training_cfg.get("test_size", 0.2)
    random_state = training_cfg.get("random_state", 42)
    models_to_train = training_cfg.get("models_to_train", ["logistic_regression"])
    models_dir = paths_cfg.get("models_dir", "models")

    os.makedirs(models_dir, exist_ok=True)

    # ── Step 1: Prepare Features ─────────────────────────────────────────────
    logger.info("► Step 1: Preparing features...")
    X, y = prepare_features(df, target_col)

    # ── Step 2: Train/Test Split ─────────────────────────────────────────────
    logger.info("► Step 2: Stratified train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    logger.info(
        f"  Train set: {len(X_train):,} samples | "
        f"Test set: {len(X_test):,} samples"
    )

    # ── Step 3: Feature Scaling ───────────────────────────────────────────────
    logger.info("► Step 3: Scaling features (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ── Step 4: Train All Models ──────────────────────────────────────────────
    logger.info(f"► Step 4: Training {len(models_to_train)} models...")
    
    model_results = {}
    
    for model_name in models_to_train:
        model_params = training_cfg.get(model_name, {})
        logger.info(f"  Training {model_name}...")
        
        result = train_single_model(
            model_name,
            model_params,
            X_train_scaled,
            y_train,
            X_test_scaled,
            y_test
        )
        
        model_results[model_name] = result
        metrics = result["metrics"]
        
        logger.info(
            f"    {model_name}: "
            f"ROC-AUC={metrics['roc_auc']:.4f}, "
            f"Accuracy={metrics['accuracy']:.4f}, "
            f"F1={metrics['f1_score']:.4f}"
        )

    # ── Step 5: Select Best Model ─────────────────────────────────────────────
    logger.info("► Step 5: Selecting best model by ROC-AUC...")
    
    best_model_name = max(
        model_results.keys(),
        key=lambda name: model_results[name]["metrics"]["roc_auc"]
    )
    
    best_result = model_results[best_model_name]
    best_model = best_result["model"]
    best_metrics = best_result["metrics"]
    
    logger.info(f"  🏆 Best model: {best_model_name}")
    logger.info("  ┌─────────────────────────────────")
    logger.info(f"  │  Model      : {best_model_name}")
    logger.info(f"  │  Accuracy   : {best_metrics['accuracy']:.4f}  ({best_metrics['accuracy']*100:.1f}%)")
    logger.info(f"  │  Precision  : {best_metrics['precision']:.4f}")
    logger.info(f"  │  Recall     : {best_metrics['recall']:.4f}")
    logger.info(f"  │  F1 Score   : {best_metrics['f1_score']:.4f}")
    logger.info(f"  │  ROC-AUC    : {best_metrics['roc_auc']:.4f}")
    logger.info("  └─────────────────────────────────")

    # ── Step 6: Generate SHAP Explainability ──────────────────────────────────
    shap_plot_path = None
    if training_cfg.get("generate_shap_plots", True):
        logger.info("► Step 6: Generating SHAP feature importance plots...")
        shap_plot_path = generate_shap_plots(
            best_model,
            pd.DataFrame(X_train_scaled, columns=X.columns),
            pd.DataFrame(X_test_scaled, columns=X.columns),
            best_model_name
        )

    # ── Step 7: Save Best Model ───────────────────────────────────────────────
    logger.info("► Step 7: Saving best model artifact...")
    version, filename = _get_next_model_version(models_dir)
    model_path = os.path.join(models_dir, filename)

    model_bundle = {
        "model": best_model,
        "model_name": best_model_name,
        "scaler": scaler,
        "feature_columns": list(X.columns),
        "target_column": target_col,
        "version": version,
        "trained_at": datetime.now().isoformat(),
        "metrics": best_metrics,
        "all_model_metrics": {
            name: res["metrics"] for name, res in model_results.items()
        },
        "model_params": training_cfg.get(best_model_name, {}),
        "training_samples": int(len(X_train)),
        "shap_plot_path": shap_plot_path
    }

    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f)

    logger.info(f"  Model saved → {model_path}  (v{version})")

    # ── Step 8: Update Metrics Registry ───────────────────────────────────────
    logger.info("► Step 8: Updating metrics registry...")
    registry = _load_metrics_registry()
    registry_entry = {
        "version": version,
        "model_file": filename,
        "model_name": best_model_name,
        "trained_at": datetime.now().isoformat(),
        "metrics": best_metrics,
        "all_models_compared": {
            name: res["metrics"]["roc_auc"] for name, res in model_results.items()
        }
    }
    registry.append(registry_entry)
    _save_metrics_registry(registry)
    logger.info(
        f"  Metrics registry updated. Total runs tracked: {len(registry)}"
    )

    results = {
        "model_path": model_path,
        "model_version": version,
        "best_model_name": best_model_name,
        "metrics": best_metrics,
        "all_model_metrics": {name: res["metrics"] for name, res in model_results.items()},
        "feature_columns": list(X.columns),
        "shap_plot_path": shap_plot_path
    }

    log_section_footer(logger, "TRAINING COMPLETE")
    return results
