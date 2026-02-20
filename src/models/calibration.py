"""
Model Calibration Module
Ensures predicted probabilities are meaningful for business decisions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
from typing import Any, Dict, Tuple, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def assess_calibration(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    Assess the calibration of a model's probability predictions.

    A well-calibrated model should have predicted probabilities that match
    actual frequencies. For example, among predictions with 70% probability,
    ~70% should actually be positive.

    Parameters
    ----------
    model : trained model
        Model with predict_proba method.
    X : np.ndarray
        Features.
    y : np.ndarray
        True labels.
    n_bins : int
        Number of bins for calibration curve.

    Returns
    -------
    Dict
        Calibration metrics and curve data.
    """
    y_proba = model.predict_proba(X)[:, 1]

    # Calculate calibration curve
    prob_true, prob_pred = calibration_curve(y, y_proba, n_bins=n_bins, strategy='uniform')

    # Calculate Brier score (lower is better)
    brier = brier_score_loss(y, y_proba)

    # Calculate Expected Calibration Error (ECE)
    bin_counts = np.histogram(y_proba, bins=n_bins, range=(0, 1))[0]
    ece = np.sum(np.abs(prob_true - prob_pred) * (bin_counts[bin_counts > 0] / len(y_proba)))

    assessment = {
        'brier_score': brier,
        'expected_calibration_error': ece,
        'is_well_calibrated': ece < 0.05,
        'calibration_curve': {
            'prob_true': prob_true.tolist(),
            'prob_pred': prob_pred.tolist()
        },
        'probability_stats': {
            'mean': y_proba.mean(),
            'std': y_proba.std(),
            'min': y_proba.min(),
            'max': y_proba.max()
        }
    }

    logger.info(f"Calibration assessment: Brier={brier:.4f}, ECE={ece:.4f}")

    return assessment


def calibrate_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str = 'sigmoid',
    cv: int = 5
) -> Any:
    """
    Calibrate a model using Platt scaling (sigmoid) or isotonic regression.

    Parameters
    ----------
    model : trained model
        Pre-trained model to calibrate.
    X_train : np.ndarray
        Training features (should be validation/held-out data for calibration).
    y_train : np.ndarray
        Training labels.
    method : str
        'sigmoid' (Platt scaling) or 'isotonic'.
    cv : int
        Cross-validation folds for calibration.

    Returns
    -------
    CalibratedClassifierCV
        Calibrated model.
    """
    logger.info(f"Calibrating model with {method} method...")

    calibrated_model = CalibratedClassifierCV(
        model,
        method=method,
        cv=cv
    )
    calibrated_model.fit(X_train, y_train)

    return calibrated_model


def compare_calibration(
    model_uncalibrated: Any,
    model_calibrated: Any,
    X: np.ndarray,
    y: np.ndarray
) -> Dict[str, Any]:
    """
    Compare calibration before and after calibration.

    Returns
    -------
    Dict
        Before/after calibration metrics.
    """
    before = assess_calibration(model_uncalibrated, X, y)
    after = assess_calibration(model_calibrated, X, y)

    comparison = {
        'before': {
            'brier_score': before['brier_score'],
            'ece': before['expected_calibration_error']
        },
        'after': {
            'brier_score': after['brier_score'],
            'ece': after['expected_calibration_error']
        },
        'improvement': {
            'brier_reduction': before['brier_score'] - after['brier_score'],
            'ece_reduction': before['expected_calibration_error'] - after['expected_calibration_error']
        },
        'calibration_needed': before['expected_calibration_error'] > 0.05
    }

    logger.info(f"Calibration comparison:")
    logger.info(f"  Before: Brier={comparison['before']['brier_score']:.4f}, "
               f"ECE={comparison['before']['ece']:.4f}")
    logger.info(f"  After:  Brier={comparison['after']['brier_score']:.4f}, "
               f"ECE={comparison['after']['ece']:.4f}")

    return comparison


def plot_calibration_curve(
    models: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    save_path: Optional[str] = None,
    n_bins: int = 10
) -> plt.Figure:
    """
    Plot calibration curves for multiple models.

    Parameters
    ----------
    models : Dict[str, model]
        Dictionary of model name to model.
    X : np.ndarray
        Features.
    y : np.ndarray
        True labels.
    save_path : str, optional
        Path to save the plot.
    n_bins : int
        Number of bins for calibration curve.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', linewidth=2)

    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for (name, model), color in zip(models.items(), colors):
        y_proba = model.predict_proba(X)[:, 1]
        prob_true, prob_pred = calibration_curve(y, y_proba, n_bins=n_bins, strategy='uniform')
        brier = brier_score_loss(y, y_proba)

        ax.plot(prob_pred, prob_true, 's-', color=color,
               label=f'{name} (Brier: {brier:.3f})', linewidth=2, markersize=8)

    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title('Calibration Curves', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved calibration plot to {save_path}")

    return fig


def plot_probability_distribution(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot distribution of predicted probabilities by actual class.

    Parameters
    ----------
    model : trained model
    X : np.ndarray
    y : np.ndarray
    save_path : str, optional

    Returns
    -------
    plt.Figure
    """
    y_proba = model.predict_proba(X)[:, 1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot histograms for each class
    ax.hist(y_proba[y == 0], bins=30, alpha=0.6, label='No Churn (Actual)',
           color='green', density=True)
    ax.hist(y_proba[y == 1], bins=30, alpha=0.6, label='Churn (Actual)',
           color='red', density=True)

    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Default Threshold')

    ax.set_xlabel('Predicted Churn Probability', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Distribution of Predicted Probabilities by Actual Class', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved probability distribution plot to {save_path}")

    return fig


if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent))

    from data.load_data import load_raw_data
    from data.preprocess import preprocess_data
    from data.feature_engineering import engineer_features
    from data.split import create_stratified_splits
    from train import prepare_features, train_models

    # Load and prepare data
    df = load_raw_data()
    df = preprocess_data(df)
    df = engineer_features(df)
    train_df, val_df, test_df = create_stratified_splits(df)

    X_train, X_val, X_test, y_train, y_val, y_test, _ = prepare_features(
        train_df, val_df, test_df
    )

    # Train models
    trained_models, _ = train_models(X_train, y_train, X_val, y_val)

    # Assess calibration of XGBoost
    print("\n=== Calibration Assessment (XGBoost) ===")
    assessment = assess_calibration(trained_models['xgboost'], X_val, y_val)
    print(f"Brier Score: {assessment['brier_score']:.4f}")
    print(f"ECE: {assessment['expected_calibration_error']:.4f}")
    print(f"Well Calibrated: {assessment['is_well_calibrated']}")

    # Calibrate if needed
    if not assessment['is_well_calibrated']:
        print("\nCalibrating model...")
        calibrated = calibrate_model(
            trained_models['xgboost'],
            X_val, y_val,
            method='sigmoid'
        )

        comparison = compare_calibration(
            trained_models['xgboost'],
            calibrated,
            X_test, y_test
        )

        print(f"\nBrier improvement: {comparison['improvement']['brier_reduction']:.4f}")
        print(f"ECE improvement: {comparison['improvement']['ece_reduction']:.4f}")
