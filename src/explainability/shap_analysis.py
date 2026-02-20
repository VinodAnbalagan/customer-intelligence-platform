"""
SHAP Analysis Module
Global and local explanations using SHAP values.
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_shap_values(
    model: Any,
    X: np.ndarray,
    feature_names: List[str],
    model_type: str = 'tree',
    sample_size: Optional[int] = None
) -> Tuple[shap.Explainer, np.ndarray]:
    """
    Compute SHAP values for a model.

    Parameters
    ----------
    model : trained model
        Model to explain.
    X : np.ndarray
        Data to compute SHAP values for.
    feature_names : list
        Names of features.
    model_type : str
        'tree' for tree-based models, 'linear' for linear models.
    sample_size : int, optional
        If provided, sample this many rows for computation.

    Returns
    -------
    Tuple[shap.Explainer, np.ndarray]
        SHAP explainer and SHAP values.
    """
    if sample_size and len(X) > sample_size:
        indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X

    logger.info(f"Computing SHAP values for {len(X_sample)} samples...")

    if model_type == 'tree':
        explainer = shap.TreeExplainer(model)
    elif model_type == 'linear':
        explainer = shap.LinearExplainer(model, X_sample)
    else:
        # Use KernelExplainer as fallback
        explainer = shap.KernelExplainer(model.predict_proba, X_sample[:100])

    shap_values = explainer.shap_values(X_sample)

    # For binary classification, take positive class SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    logger.info(f"SHAP values computed. Shape: {shap_values.shape}")

    return explainer, shap_values


def get_global_importance(
    shap_values: np.ndarray,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Calculate global feature importance from SHAP values.

    Returns
    -------
    pd.DataFrame
        Features sorted by mean absolute SHAP value.
    """
    importance = np.abs(shap_values).mean(axis=0)

    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })

    return df.sort_values('importance', ascending=False).reset_index(drop=True)


def plot_shap_summary(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: List[str],
    save_path: Optional[str] = None,
    max_display: int = 20
) -> plt.Figure:
    """
    Create SHAP summary plot (beeswarm).

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values.
    X : np.ndarray
        Feature values.
    feature_names : list
        Feature names.
    save_path : str, optional
        Path to save the plot.
    max_display : int
        Maximum features to display.

    Returns
    -------
    plt.Figure
    """
    plt.figure(figsize=(10, 8))

    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        max_display=max_display,
        show=False
    )

    plt.title('SHAP Feature Importance (Impact on Churn Prediction)', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved SHAP summary plot to {save_path}")

    return plt.gcf()


def plot_shap_bar(
    shap_values: np.ndarray,
    feature_names: List[str],
    save_path: Optional[str] = None,
    max_display: int = 15
) -> plt.Figure:
    """
    Create SHAP bar plot (mean absolute values).

    Returns
    -------
    plt.Figure
    """
    importance = get_global_importance(shap_values, feature_names)

    fig, ax = plt.subplots(figsize=(10, 8))

    top_features = importance.head(max_display)

    colors = plt.cm.Reds(np.linspace(0.3, 0.8, len(top_features)))
    bars = ax.barh(
        range(len(top_features)),
        top_features['importance'].values,
        color=colors[::-1]
    )

    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values)
    ax.invert_yaxis()
    ax.set_xlabel('Mean |SHAP Value|', fontsize=12)
    ax.set_title('Feature Importance (Mean Absolute SHAP Value)', fontsize=14)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved SHAP bar plot to {save_path}")

    return fig


def plot_shap_waterfall(
    explainer: shap.Explainer,
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: List[str],
    index: int,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create SHAP waterfall plot for a single prediction.

    Parameters
    ----------
    explainer : shap.Explainer
    shap_values : np.ndarray
    X : np.ndarray
    feature_names : list
    index : int
        Index of the sample to explain.
    save_path : str, optional

    Returns
    -------
    plt.Figure
    """
    # Create Explanation object
    if hasattr(explainer, 'expected_value'):
        expected_value = explainer.expected_value
        if isinstance(expected_value, np.ndarray):
            expected_value = expected_value[1]  # Positive class for binary
    else:
        expected_value = 0

    explanation = shap.Explanation(
        values=shap_values[index],
        base_values=expected_value,
        data=X[index],
        feature_names=feature_names
    )

    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(explanation, show=False, max_display=15)
    plt.title(f'SHAP Explanation for Sample {index}', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved SHAP waterfall plot to {save_path}")

    return plt.gcf()


def get_customer_explanation(
    explainer: shap.Explainer,
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: List[str],
    index: int,
    top_n: int = 5
) -> Dict:
    """
    Get human-readable explanation for a single customer.

    Returns
    -------
    Dict
        Explanation with top positive and negative factors.
    """
    if hasattr(explainer, 'expected_value'):
        base_value = explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[1]
    else:
        base_value = 0

    customer_shap = shap_values[index]
    customer_features = X[index]

    # Create feature-value-shap pairs
    pairs = list(zip(feature_names, customer_features, customer_shap))

    # Sort by SHAP value (most positive = increases churn risk)
    pairs_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)

    # Get top factors increasing churn risk
    increasing_risk = [(name, val, shap_val) for name, val, shap_val in pairs_sorted if shap_val > 0][:top_n]

    # Get top factors decreasing churn risk
    decreasing_risk = [(name, val, shap_val) for name, val, shap_val in pairs_sorted if shap_val < 0][-top_n:]
    decreasing_risk.reverse()

    predicted_prob = base_value + sum(customer_shap)
    # Convert from log-odds to probability if needed
    if predicted_prob < 0 or predicted_prob > 1:
        predicted_prob = 1 / (1 + np.exp(-predicted_prob))

    explanation = {
        'base_probability': base_value,
        'predicted_probability': float(predicted_prob),
        'top_risk_factors': [
            {'feature': name, 'value': float(val), 'impact': float(shap_val)}
            for name, val, shap_val in increasing_risk
        ],
        'top_protective_factors': [
            {'feature': name, 'value': float(val), 'impact': float(abs(shap_val))}
            for name, val, shap_val in decreasing_risk
        ]
    }

    return explanation


def find_example_customers(
    model: Any,
    X: np.ndarray,
    y: np.ndarray
) -> Dict[str, int]:
    """
    Find example customers for different risk levels.

    Returns
    -------
    Dict
        Indices for high-risk, low-risk, and borderline customers.
    """
    y_proba = model.predict_proba(X)[:, 1]

    # High risk: predicted > 0.8, actually churned
    high_risk_mask = (y_proba > 0.8) & (y == 1)
    high_risk_idx = np.where(high_risk_mask)[0]
    high_risk = int(high_risk_idx[0]) if len(high_risk_idx) > 0 else None

    # Low risk: predicted < 0.2, didn't churn
    low_risk_mask = (y_proba < 0.2) & (y == 0)
    low_risk_idx = np.where(low_risk_mask)[0]
    low_risk = int(low_risk_idx[0]) if len(low_risk_idx) > 0 else None

    # Borderline: predicted ~0.5
    borderline_mask = (y_proba > 0.4) & (y_proba < 0.6)
    borderline_idx = np.where(borderline_mask)[0]
    borderline = int(borderline_idx[0]) if len(borderline_idx) > 0 else None

    return {
        'high_risk': high_risk,
        'low_risk': low_risk,
        'borderline': borderline
    }


if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent))

    from data.load_data import load_raw_data
    from data.preprocess import preprocess_data
    from data.feature_engineering import engineer_features
    from data.split import create_stratified_splits
    from models.train import prepare_features, train_models

    # Load and prepare data
    df = load_raw_data()
    df = preprocess_data(df)
    df = engineer_features(df)
    train_df, val_df, test_df = create_stratified_splits(df)

    X_train, X_val, X_test, y_train, y_val, y_test, preprocessors = prepare_features(
        train_df, val_df, test_df
    )

    # Train models
    trained_models, _ = train_models(X_train, y_train, X_val, y_val)

    # Compute SHAP values for XGBoost
    print("\n=== Computing SHAP Values ===")
    explainer, shap_values = compute_shap_values(
        trained_models['xgboost'],
        X_val,
        preprocessors['feature_cols'],
        model_type='tree',
        sample_size=500
    )

    # Global importance
    print("\n=== Global Feature Importance (SHAP) ===")
    importance = get_global_importance(shap_values, preprocessors['feature_cols'])
    print(importance.head(10))

    # Find example customers
    examples = find_example_customers(trained_models['xgboost'], X_val, y_val)
    print(f"\n=== Example Customer Indices ===")
    print(examples)

    # Get explanation for high-risk customer
    if examples['high_risk'] is not None:
        explanation = get_customer_explanation(
            explainer, shap_values, X_val,
            preprocessors['feature_cols'],
            examples['high_risk']
        )
        print(f"\n=== High-Risk Customer Explanation ===")
        print(f"Predicted churn probability: {explanation['predicted_probability']:.2%}")
        print("Top risk factors:")
        for factor in explanation['top_risk_factors']:
            print(f"  - {factor['feature']}: impact +{factor['impact']:.3f}")
