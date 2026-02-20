"""
Imbalance Handling Module
Compares different strategies for handling class imbalance in churn prediction.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import xgboost as xgb
from typing import Dict, Tuple, Any, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_imbalance_strategies() -> List[str]:
    """Return list of available imbalance handling strategies."""
    return [
        'none',
        'class_weight',
        'smote',
        'smote_tomek',
        'random_undersample'
    ]


def apply_resampling(
    X_train: np.ndarray,
    y_train: np.ndarray,
    strategy: str,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a resampling strategy to training data.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    strategy : str
        One of: 'none', 'smote', 'smote_tomek', 'random_undersample'
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Resampled X_train, y_train.
    """
    if strategy == 'none' or strategy == 'class_weight':
        return X_train, y_train

    if strategy == 'smote':
        sampler = SMOTE(random_state=random_state)
    elif strategy == 'smote_tomek':
        sampler = SMOTETomek(random_state=random_state)
    elif strategy == 'random_undersample':
        sampler = RandomUnderSampler(random_state=random_state)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

    logger.info(f"Resampled with {strategy}: {len(y_train)} -> {len(y_resampled)} samples")
    logger.info(f"  Class distribution: {np.bincount(y_resampled.astype(int))}")

    return X_resampled, y_resampled


def train_with_strategy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    strategy: str,
    random_state: int = 42
) -> Tuple[Any, Dict[str, float]]:
    """
    Train XGBoost model with a specific imbalance handling strategy.

    Returns
    -------
    Tuple[model, Dict]
        Trained model and validation metrics.
    """
    # Apply resampling (if applicable)
    X_train_resampled, y_train_resampled = apply_resampling(
        X_train, y_train, strategy, random_state
    )

    # Configure model based on strategy
    if strategy == 'class_weight':
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    else:
        scale_pos_weight = 1

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    model.fit(X_train_resampled, y_train_resampled, verbose=False)

    # Evaluate
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    metrics = {
        'f1': f1_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'roc_auc': roc_auc_score(y_val, y_proba),
        'train_samples': len(y_train_resampled)
    }

    return model, metrics


def compare_imbalance_strategies(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    strategies: List[str] = None,
    verbose: bool = True
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]], str]:
    """
    Compare all imbalance handling strategies.

    Parameters
    ----------
    X_train, y_train : training data
    X_val, y_val : validation data
    strategies : list, optional
        Strategies to compare. If None, uses all.
    verbose : bool
        If True, log progress.

    Returns
    -------
    Tuple[Dict, Dict, str]
        (trained_models, all_metrics, best_strategy)
    """
    if strategies is None:
        strategies = get_imbalance_strategies()

    if verbose:
        logger.info(f"Comparing {len(strategies)} imbalance handling strategies...")
        logger.info(f"Original class distribution: {np.bincount(y_train.astype(int))}")

    all_models = {}
    all_metrics = {}

    for strategy in strategies:
        if verbose:
            logger.info(f"\nTesting strategy: {strategy}")

        model, metrics = train_with_strategy(
            X_train, y_train, X_val, y_val, strategy
        )

        all_models[strategy] = model
        all_metrics[strategy] = metrics

        if verbose:
            logger.info(f"  F1: {metrics['f1']:.4f}, "
                       f"Precision: {metrics['precision']:.4f}, "
                       f"Recall: {metrics['recall']:.4f}")

    # Find best strategy (by F1 score)
    best_strategy = max(all_metrics.keys(), key=lambda k: all_metrics[k]['f1'])

    if verbose:
        logger.info(f"\nBest strategy: {best_strategy} (F1: {all_metrics[best_strategy]['f1']:.4f})")

    return all_models, all_metrics, best_strategy


def get_comparison_table(all_metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Create a comparison table of all strategies.

    Returns
    -------
    pd.DataFrame
        Comparison table sorted by F1 score.
    """
    df = pd.DataFrame(all_metrics).T
    df = df.round(4)
    df = df.sort_values('f1', ascending=False)
    return df


def get_strategy_recommendation(all_metrics: Dict[str, Dict[str, float]]) -> Dict:
    """
    Generate a recommendation based on comparison results.

    Returns
    -------
    dict
        Recommendation with rationale.
    """
    df = get_comparison_table(all_metrics)
    best = df.index[0]

    recommendation = {
        'recommended_strategy': best,
        'f1_score': all_metrics[best]['f1'],
        'rationale': '',
        'trade_offs': {}
    }

    # Add rationale based on strategy
    rationales = {
        'class_weight': (
            "Class weighting is simple, effective, and doesn't alter training data. "
            "It's the most practical choice for production systems."
        ),
        'smote': (
            "SMOTE creates synthetic minority samples, improving recall. "
            "However, it increases training time and may introduce noise."
        ),
        'smote_tomek': (
            "SMOTE + Tomek cleaning balances the dataset while removing "
            "ambiguous samples near the decision boundary."
        ),
        'random_undersample': (
            "Undersampling is fast but discards majority class information. "
            "Use only when training data is abundant."
        ),
        'none': (
            "No imbalance handling works best, suggesting the model naturally "
            "handles the class distribution. Monitor for bias toward majority class."
        )
    }

    recommendation['rationale'] = rationales.get(best, "Strategy selected based on F1 score.")

    # Add trade-offs
    for strategy, metrics in all_metrics.items():
        if strategy != best:
            f1_diff = all_metrics[best]['f1'] - metrics['f1']
            recommendation['trade_offs'][strategy] = {
                'f1_difference': f1_diff,
                'precision': metrics['precision'],
                'recall': metrics['recall']
            }

    return recommendation


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))

    from data.load_data import load_raw_data
    from data.preprocess import preprocess_data
    from data.feature_engineering import engineer_features
    from data.split import create_stratified_splits
    from train import prepare_features

    # Load and prepare data
    df = load_raw_data()
    df = preprocess_data(df)
    df = engineer_features(df)
    train_df, val_df, test_df = create_stratified_splits(df)

    X_train, X_val, X_test, y_train, y_val, y_test, _ = prepare_features(
        train_df, val_df, test_df
    )

    # Compare strategies
    all_models, all_metrics, best_strategy = compare_imbalance_strategies(
        X_train, y_train, X_val, y_val
    )

    # Print comparison table
    print("\n=== Imbalance Strategy Comparison ===")
    print(get_comparison_table(all_metrics))

    # Get recommendation
    print("\n=== Recommendation ===")
    rec = get_strategy_recommendation(all_metrics)
    print(f"Recommended: {rec['recommended_strategy']}")
    print(f"Rationale: {rec['rationale']}")
