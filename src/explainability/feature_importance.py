"""
Feature Importance Comparison Module
Compares feature importance across different methods.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from typing import Any, Dict, List, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_logistic_importance(
    model: Any,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Get feature importance from logistic regression coefficients.

    Returns
    -------
    pd.DataFrame
        Features sorted by absolute coefficient value.
    """
    importance = np.abs(model.coef_[0])

    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance,
        'method': 'logistic_coefficients'
    })

    return df.sort_values('importance', ascending=False).reset_index(drop=True)


def get_tree_importance(
    model: Any,
    feature_names: List[str],
    model_name: str
) -> pd.DataFrame:
    """
    Get feature importance from tree-based models (impurity or gain).

    Returns
    -------
    pd.DataFrame
        Features sorted by importance.
    """
    importance = model.feature_importances_

    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance,
        'method': f'{model_name}_builtin'
    })

    return df.sort_values('importance', ascending=False).reset_index(drop=True)


def get_permutation_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 10,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Calculate permutation importance.

    Permutation importance measures how much model performance degrades
    when a feature's values are randomly shuffled.

    Returns
    -------
    pd.DataFrame
        Features sorted by permutation importance.
    """
    logger.info("Computing permutation importance...")

    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )

    df = pd.DataFrame({
        'feature': feature_names,
        'importance': result.importances_mean,
        'std': result.importances_std,
        'method': 'permutation'
    })

    return df.sort_values('importance', ascending=False).reset_index(drop=True)


def compare_feature_importance(
    models: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    shap_importance: Optional[pd.DataFrame] = None,
    top_n: int = 15
) -> pd.DataFrame:
    """
    Compare feature importance across all methods.

    Parameters
    ----------
    models : Dict[str, model]
        Dictionary of model name to trained model.
    X, y : data for permutation importance
    feature_names : list
    shap_importance : pd.DataFrame, optional
        Pre-computed SHAP importance.
    top_n : int
        Number of top features to include.

    Returns
    -------
    pd.DataFrame
        Combined importance rankings.
    """
    all_importance = []

    # Logistic regression coefficients
    if 'logistic_regression' in models:
        lr_imp = get_logistic_importance(models['logistic_regression'], feature_names)
        all_importance.append(lr_imp)

    # Tree-based importances
    for name in ['random_forest', 'xgboost', 'lightgbm']:
        if name in models:
            tree_imp = get_tree_importance(models[name], feature_names, name)
            all_importance.append(tree_imp)

    # Permutation importance (use best tree model)
    best_tree = None
    for name in ['xgboost', 'lightgbm', 'random_forest']:
        if name in models:
            best_tree = models[name]
            break

    if best_tree is not None:
        perm_imp = get_permutation_importance(best_tree, X, y, feature_names)
        all_importance.append(perm_imp)

    # Add SHAP importance if provided
    if shap_importance is not None:
        shap_df = shap_importance.copy()
        shap_df['method'] = 'shap'
        all_importance.append(shap_df)

    # Combine all
    combined = pd.concat(all_importance, ignore_index=True)

    # Create pivot table for comparison
    pivot = combined.pivot_table(
        values='importance',
        index='feature',
        columns='method',
        aggfunc='first'
    )

    # Normalize each column to [0, 1] for fair comparison
    for col in pivot.columns:
        pivot[col] = pivot[col] / pivot[col].max()

    # Add average rank
    ranks = pivot.rank(ascending=False)
    pivot['avg_rank'] = ranks.mean(axis=1)

    # Sort by average rank
    pivot = pivot.sort_values('avg_rank')

    return pivot.head(top_n)


def plot_importance_comparison(
    comparison_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot feature importance comparison across methods.

    Returns
    -------
    plt.Figure
    """
    # Remove avg_rank for plotting
    plot_data = comparison_df.drop(columns=['avg_rank'], errors='ignore')

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create grouped bar chart
    x = np.arange(len(plot_data))
    width = 0.15
    methods = plot_data.columns.tolist()

    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

    for i, (method, color) in enumerate(zip(methods, colors)):
        offset = (i - len(methods)/2 + 0.5) * width
        ax.barh(x + offset, plot_data[method].values, width,
               label=method.replace('_', ' ').title(), color=color)

    ax.set_yticks(x)
    ax.set_yticklabels(plot_data.index)
    ax.invert_yaxis()
    ax.set_xlabel('Normalized Importance', fontsize=12)
    ax.set_title('Feature Importance Comparison Across Methods', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved importance comparison plot to {save_path}")

    return fig


def get_consensus_features(
    comparison_df: pd.DataFrame,
    top_n: int = 10
) -> List[str]:
    """
    Get features that rank highly across all methods (consensus).

    Returns
    -------
    List[str]
        List of consensus top features.
    """
    return comparison_df.head(top_n).index.tolist()


def get_importance_summary(
    comparison_df: pd.DataFrame
) -> Dict:
    """
    Generate a summary of feature importance analysis.

    Returns
    -------
    Dict
        Summary with insights.
    """
    top_features = comparison_df.head(5).index.tolist()

    summary = {
        'top_5_features': top_features,
        'methods_compared': [
            col for col in comparison_df.columns if col != 'avg_rank'
        ],
        'most_consistent_feature': comparison_df.index[0],
        'insight': (
            f"The most important features for predicting churn are: "
            f"{', '.join(top_features[:3])}. These features consistently rank "
            f"highly across all importance methods, indicating robust predictive power."
        )
    }

    return summary


if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent))

    from data.load_data import load_raw_data
    from data.preprocess import preprocess_data
    from data.feature_engineering import engineer_features
    from data.split import create_stratified_splits
    from models.train import prepare_features, train_models
    from shap_analysis import compute_shap_values, get_global_importance

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

    # Get SHAP importance
    _, shap_values = compute_shap_values(
        trained_models['xgboost'],
        X_val,
        preprocessors['feature_cols'],
        sample_size=500
    )
    shap_imp = get_global_importance(shap_values, preprocessors['feature_cols'])

    # Compare all methods
    print("\n=== Feature Importance Comparison ===")
    comparison = compare_feature_importance(
        trained_models,
        X_val, y_val,
        preprocessors['feature_cols'],
        shap_importance=shap_imp
    )
    print(comparison)

    # Get summary
    summary = get_importance_summary(comparison)
    print(f"\n=== Summary ===")
    print(f"Top 5 features: {summary['top_5_features']}")
    print(f"Insight: {summary['insight']}")
