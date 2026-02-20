"""
MLflow Experiment Tracking Module
Tracks model experiments, parameters, metrics, and artifacts.
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss,
    confusion_matrix, roc_curve, precision_recall_curve
)
from typing import Any, Dict, Optional, List
from pathlib import Path
import logging
import tempfile
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_experiment(
    experiment_name: str = "churn_prediction",
    tracking_uri: Optional[str] = None
) -> str:
    """
    Set up MLflow experiment.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.
    tracking_uri : str, optional
        MLflow tracking URI. If None, uses local mlruns directory.

    Returns
    -------
    str
        Experiment ID.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        # Use local mlruns directory
        project_root = Path(__file__).parent.parent.parent
        mlruns_path = project_root / "mlruns"
        mlflow.set_tracking_uri(f"file://{mlruns_path}")

    # Create or get experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        logger.info(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        logger.info(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")

    mlflow.set_experiment(experiment_name)

    return experiment_id


def log_model_run(
    model: Any,
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Dict,
    feature_names: List[str],
    tags: Optional[Dict] = None,
    artifact_dir: Optional[str] = None
) -> str:
    """
    Log a complete model run to MLflow.

    Parameters
    ----------
    model : trained model
    model_name : str
    X_train, y_train : training data
    X_val, y_val : validation data
    params : dict
        Model hyperparameters.
    feature_names : list
    tags : dict, optional
        Additional tags for the run.
    artifact_dir : str, optional
        Directory to save artifacts.

    Returns
    -------
    str
        Run ID.
    """
    with mlflow.start_run(run_name=model_name) as run:
        run_id = run.info.run_id
        logger.info(f"Starting MLflow run: {run_id}")

        # Log parameters
        mlflow.log_params(params)

        # Calculate metrics
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred),
            'f1_macro': f1_score(y_val, y_pred, average='macro'),
            'roc_auc': roc_auc_score(y_val, y_proba),
            'pr_auc': average_precision_score(y_val, y_proba),
            'log_loss': log_loss(y_val, y_proba)
        }

        # Log metrics
        mlflow.log_metrics(metrics)
        logger.info(f"Logged metrics: F1={metrics['f1']:.4f}, AUC={metrics['roc_auc']:.4f}")

        # Log tags
        if tags:
            mlflow.set_tags(tags)
        mlflow.set_tag('model_type', model_name)

        # Create and log artifacts
        with tempfile.TemporaryDirectory() as tmpdir:
            # Confusion matrix
            cm_path = _plot_confusion_matrix(y_val, y_pred, tmpdir)
            mlflow.log_artifact(cm_path)

            # ROC curve
            roc_path = _plot_roc_curve(y_val, y_proba, tmpdir)
            mlflow.log_artifact(roc_path)

            # PR curve
            pr_path = _plot_pr_curve(y_val, y_proba, tmpdir)
            mlflow.log_artifact(pr_path)

            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                fi_path = _plot_feature_importance(
                    model.feature_importances_,
                    feature_names,
                    tmpdir
                )
                mlflow.log_artifact(fi_path)

        # Log model
        if 'xgboost' in model_name.lower():
            mlflow.xgboost.log_model(model, "model")
        elif 'lightgbm' in model_name.lower():
            mlflow.lightgbm.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")

        logger.info(f"Run completed: {run_id}")

        return run_id


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_dir: str
) -> str:
    """Create and save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['No Churn', 'Churn'])
    ax.set_yticklabels(['No Churn', 'Churn'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')

    # Add text annotations
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                   color='white' if cm[i, j] > cm.max()/2 else 'black',
                   fontsize=14)

    plt.tight_layout()

    path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(path, dpi=150)
    plt.close()

    return path


def _plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_dir: str
) -> str:
    """Create and save ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    path = os.path.join(save_dir, 'roc_curve.png')
    plt.savefig(path, dpi=150)
    plt.close()

    return path


def _plot_pr_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_dir: str
) -> str:
    """Create and save Precision-Recall curve plot."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    auc = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(recall, precision, 'b-', linewidth=2, label=f'PR (AUC = {auc:.3f})')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    path = os.path.join(save_dir, 'pr_curve.png')
    plt.savefig(path, dpi=150)
    plt.close()

    return path


def _plot_feature_importance(
    importances: np.ndarray,
    feature_names: List[str],
    save_dir: str,
    top_n: int = 15
) -> str:
    """Create and save feature importance plot."""
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.barh(range(top_n), importances[indices][::-1],
           color=plt.cm.Blues(np.linspace(0.3, 0.8, top_n)))
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices[::-1]])
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    path = os.path.join(save_dir, 'feature_importance.png')
    plt.savefig(path, dpi=150)
    plt.close()

    return path


def get_best_run(
    experiment_name: str = "churn_prediction",
    metric: str = "f1"
) -> Dict:
    """
    Get the best run from an experiment.

    Returns
    -------
    Dict
        Information about the best run.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"]
    )

    if len(runs) == 0:
        raise ValueError("No runs found in experiment")

    best_run = runs.iloc[0]

    return {
        'run_id': best_run['run_id'],
        'model_type': best_run.get('tags.model_type', 'unknown'),
        'metrics': {
            col.replace('metrics.', ''): best_run[col]
            for col in runs.columns if col.startswith('metrics.')
        },
        'params': {
            col.replace('params.', ''): best_run[col]
            for col in runs.columns if col.startswith('params.')
        }
    }


if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent))

    from data.load_data import load_raw_data
    from data.preprocess import preprocess_data
    from data.feature_engineering import engineer_features
    from data.split import create_stratified_splits
    from models.train import prepare_features, get_models

    # Load and prepare data
    df = load_raw_data()
    df = preprocess_data(df)
    df = engineer_features(df)
    train_df, val_df, test_df = create_stratified_splits(df)

    X_train, X_val, X_test, y_train, y_val, y_test, preprocessors = prepare_features(
        train_df, val_df, test_df
    )

    # Setup experiment
    setup_experiment("churn_prediction")

    # Train and log XGBoost
    import xgboost as xgb

    params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1
    }

    model = xgb.XGBClassifier(**params, random_state=42)
    model.fit(X_train, y_train)

    run_id = log_model_run(
        model,
        'xgboost',
        X_train, y_train,
        X_val, y_val,
        params,
        preprocessors['feature_cols'],
        tags={'imbalance_strategy': 'none'}
    )

    print(f"\nLogged run: {run_id}")

    # Get best run
    best = get_best_run()
    print(f"\nBest run: {best['run_id']}")
    print(f"Model: {best['model_type']}")
    print(f"F1: {best['metrics'].get('f1', 'N/A'):.4f}")
