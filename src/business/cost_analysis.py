"""
Business Cost Analysis Module
Computes optimal thresholds and ROI based on business costs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default business cost assumptions (documented)
DEFAULT_COSTS = {
    'cost_of_churn': 500,      # Revenue lost when customer churns
    'cost_of_retention': 100,   # Cost of retention campaign/offer
    'cost_fp': 100,             # Wasted retention offer (false positive)
    'cost_fn': 500,             # Lost customer (false negative)
}


def compute_threshold_costs(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: np.ndarray = None,
    costs: Dict = None
) -> pd.DataFrame:
    """
    Compute business costs for different prediction thresholds.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_proba : np.ndarray
        Predicted probabilities.
    thresholds : np.ndarray, optional
        Thresholds to evaluate. Default: 0.1 to 0.9 by 0.05.
    costs : Dict, optional
        Cost parameters. Uses defaults if not provided.

    Returns
    -------
    pd.DataFrame
        Costs and metrics for each threshold.
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05)

    if costs is None:
        costs = DEFAULT_COSTS

    results = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)

        tn, fp, fn, tp = cm.ravel()

        # Calculate costs
        total_cost = (fn * costs['cost_fn']) + (fp * costs['cost_fp'])
        potential_savings = tp * (costs['cost_of_churn'] - costs['cost_of_retention'])
        net_savings = potential_savings - (fp * costs['cost_of_retention'])

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results.append({
            'threshold': threshold,
            'TP': tp,
            'FP': fp,
            'TN': tn,
            'FN': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'total_cost': total_cost,
            'potential_savings': potential_savings,
            'net_savings': net_savings,
            'customers_flagged': tp + fp,
            'customers_flagged_pct': (tp + fp) / len(y_true) * 100
        })

    return pd.DataFrame(results)


def compute_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    costs: Dict = None,
    optimize_for: str = 'net_savings'
) -> Dict:
    """
    Find the optimal prediction threshold for business objectives.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_proba : np.ndarray
        Predicted probabilities.
    costs : Dict, optional
        Cost parameters.
    optimize_for : str
        Metric to optimize: 'net_savings', 'total_cost', 'f1'.

    Returns
    -------
    Dict
        Optimal threshold and associated metrics.
    """
    if costs is None:
        costs = DEFAULT_COSTS

    df = compute_threshold_costs(y_true, y_proba, costs=costs)

    if optimize_for == 'net_savings':
        best_idx = df['net_savings'].idxmax()
    elif optimize_for == 'total_cost':
        best_idx = df['total_cost'].idxmin()
    elif optimize_for == 'f1':
        best_idx = df['f1'].idxmax()
    else:
        raise ValueError(f"Unknown optimization target: {optimize_for}")

    best_row = df.loc[best_idx]

    result = {
        'optimal_threshold': best_row['threshold'],
        'metrics': {
            'precision': best_row['precision'],
            'recall': best_row['recall'],
            'f1': best_row['f1'],
            'customers_flagged': int(best_row['customers_flagged']),
            'customers_flagged_pct': best_row['customers_flagged_pct']
        },
        'business_impact': {
            'total_cost': best_row['total_cost'],
            'potential_savings': best_row['potential_savings'],
            'net_savings': best_row['net_savings'],
            'true_positives': int(best_row['TP']),
            'false_positives': int(best_row['FP']),
            'false_negatives': int(best_row['FN'])
        },
        'cost_assumptions': costs,
        'all_thresholds': df
    }

    logger.info(f"Optimal threshold: {result['optimal_threshold']:.2f} "
               f"(Net savings: ${result['business_impact']['net_savings']:,.0f})")

    return result


def calculate_business_impact(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.5,
    costs: Dict = None
) -> Dict:
    """
    Calculate the business impact of using the model.

    Returns
    -------
    Dict
        Business impact metrics and comparisons.
    """
    if costs is None:
        costs = DEFAULT_COSTS

    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # Current model performance
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Calculate with model
    with_model = {
        'churners_caught': tp,
        'churners_missed': fn,
        'false_alarms': fp,
        'retention_cost': (tp + fp) * costs['cost_of_retention'],
        'lost_revenue': fn * costs['cost_of_churn'],
        'saved_revenue': tp * costs['cost_of_churn'],
        'net_impact': (tp * costs['cost_of_churn']) - ((tp + fp) * costs['cost_of_retention']) - (fn * costs['cost_of_churn'])
    }

    # Calculate without model (treat everyone same)
    total_churners = int(y.sum())
    total_customers = len(y)

    # Option 1: No intervention
    no_intervention = {
        'lost_revenue': total_churners * costs['cost_of_churn'],
        'retention_cost': 0,
        'net_impact': -total_churners * costs['cost_of_churn']
    }

    # Option 2: Intervene on everyone
    intervene_all = {
        'lost_revenue': 0,  # Assume 100% retention if we intervene
        'retention_cost': total_customers * costs['cost_of_retention'],
        'net_impact': (total_churners * costs['cost_of_churn']) - (total_customers * costs['cost_of_retention'])
    }

    # Comparison
    savings_vs_no_action = with_model['net_impact'] - no_intervention['net_impact']
    savings_vs_intervene_all = with_model['net_impact'] - intervene_all['net_impact']

    return {
        'with_model': with_model,
        'no_intervention': no_intervention,
        'intervene_all': intervene_all,
        'comparison': {
            'savings_vs_no_action': savings_vs_no_action,
            'savings_vs_intervene_all': savings_vs_intervene_all,
            'best_strategy': 'model' if with_model['net_impact'] > max(
                no_intervention['net_impact'], intervene_all['net_impact']
            ) else 'other'
        },
        'summary': (
            f"Using the model at threshold {threshold:.2f} saves "
            f"${savings_vs_no_action:,.0f} compared to no action."
        )
    }


def plot_cost_curve(
    threshold_df: pd.DataFrame,
    optimal_threshold: float,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot cost curve showing costs at different thresholds.

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Total cost and net savings
    ax1 = axes[0]
    ax1.plot(threshold_df['threshold'], threshold_df['total_cost'],
            'r-', linewidth=2, label='Total Cost (FP + FN)')
    ax1.plot(threshold_df['threshold'], threshold_df['net_savings'],
            'g-', linewidth=2, label='Net Savings')
    ax1.axvline(x=optimal_threshold, color='blue', linestyle='--',
               linewidth=2, label=f'Optimal ({optimal_threshold:.2f})')
    ax1.set_xlabel('Prediction Threshold', fontsize=12)
    ax1.set_ylabel('$ Amount', fontsize=12)
    ax1.set_title('Business Cost Analysis', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Precision/Recall trade-off
    ax2 = axes[1]
    ax2.plot(threshold_df['threshold'], threshold_df['precision'],
            'b-', linewidth=2, label='Precision')
    ax2.plot(threshold_df['threshold'], threshold_df['recall'],
            'orange', linewidth=2, label='Recall')
    ax2.plot(threshold_df['threshold'], threshold_df['f1'],
            'g-', linewidth=2, label='F1 Score')
    ax2.axvline(x=optimal_threshold, color='red', linestyle='--',
               linewidth=2, label=f'Optimal ({optimal_threshold:.2f})')
    ax2.set_xlabel('Prediction Threshold', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Precision-Recall Trade-off', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved cost curve to {save_path}")

    return fig


def generate_cost_report(
    impact: Dict,
    optimal: Dict
) -> str:
    """
    Generate a human-readable cost analysis report.

    Returns
    -------
    str
        Formatted report.
    """
    report = """
# Business Cost Analysis Report

## Summary

{summary}

## Cost Assumptions

| Cost Type | Amount |
|-----------|--------|
| Cost of losing a customer | ${cost_churn:,} |
| Cost of retention campaign | ${cost_retention:,} |
| Cost of false positive | ${cost_fp:,} |
| Cost of false negative | ${cost_fn:,} |

## Optimal Threshold: {threshold:.2f}

At this threshold:
- **Churners caught:** {tp:,} out of {total_churners:,} ({recall:.1%} recall)
- **False alarms:** {fp:,} customers flagged incorrectly
- **Churners missed:** {fn:,} customers not caught

## Financial Impact

| Strategy | Net Impact |
|----------|------------|
| **Use Model (Recommended)** | ${with_model:,} |
| No intervention | ${no_action:,} |
| Intervene on everyone | ${all:,} |

**Model savings vs no action:** ${savings:,}

## Recommendation

{recommendation}
""".format(
        summary=impact.get('summary', ''),
        cost_churn=optimal['cost_assumptions']['cost_of_churn'],
        cost_retention=optimal['cost_assumptions']['cost_of_retention'],
        cost_fp=optimal['cost_assumptions']['cost_fp'],
        cost_fn=optimal['cost_assumptions']['cost_fn'],
        threshold=optimal['optimal_threshold'],
        tp=optimal['business_impact']['true_positives'],
        total_churners=optimal['business_impact']['true_positives'] + optimal['business_impact']['false_negatives'],
        recall=optimal['metrics']['recall'],
        fp=optimal['business_impact']['false_positives'],
        fn=optimal['business_impact']['false_negatives'],
        with_model=impact['with_model']['net_impact'],
        no_action=impact['no_intervention']['net_impact'],
        all=impact['intervene_all']['net_impact'],
        savings=impact['comparison']['savings_vs_no_action'],
        recommendation=(
            "Deploy the model with the optimal threshold for maximum ROI. "
            "Focus retention efforts on customers flagged as high-risk."
        )
    )

    return report


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

    # Get predictions
    y_proba = trained_models['xgboost'].predict_proba(X_test)[:, 1]

    # Find optimal threshold
    optimal = compute_optimal_threshold(y_test, y_proba)
    print(f"\n=== Optimal Threshold ===")
    print(f"Threshold: {optimal['optimal_threshold']:.2f}")
    print(f"Net Savings: ${optimal['business_impact']['net_savings']:,.0f}")
    print(f"Precision: {optimal['metrics']['precision']:.2%}")
    print(f"Recall: {optimal['metrics']['recall']:.2%}")

    # Calculate business impact
    impact = calculate_business_impact(
        trained_models['xgboost'],
        X_test, y_test,
        threshold=optimal['optimal_threshold']
    )
    print(f"\n{impact['summary']}")
