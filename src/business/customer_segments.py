"""
Customer Segmentation Module
Creates actionable customer risk segments based on model predictions.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Risk segment definitions
SEGMENT_DEFINITIONS = {
    'safe': {
        'threshold_low': 0.0,
        'threshold_high': 0.2,
        'name': 'Safe',
        'description': 'Low churn risk - no immediate action needed',
        'recommended_action': 'Standard engagement, monitor for changes',
        'intervention_priority': 4
    },
    'monitor': {
        'threshold_low': 0.2,
        'threshold_high': 0.5,
        'name': 'Monitor',
        'description': 'Moderate churn risk - light touch engagement',
        'recommended_action': 'Proactive check-ins, satisfaction surveys',
        'intervention_priority': 3
    },
    'at_risk': {
        'threshold_low': 0.5,
        'threshold_high': 0.8,
        'name': 'At Risk',
        'description': 'High churn risk - proactive retention needed',
        'recommended_action': 'Targeted retention campaigns, special offers',
        'intervention_priority': 2
    },
    'critical': {
        'threshold_low': 0.8,
        'threshold_high': 1.0,
        'name': 'Critical',
        'description': 'Very high churn risk - immediate intervention',
        'recommended_action': 'Personal outreach, significant incentives',
        'intervention_priority': 1
    }
}


def create_risk_segments(
    y_proba: np.ndarray,
    segment_definitions: Dict = None
) -> np.ndarray:
    """
    Assign customers to risk segments based on churn probability.

    Parameters
    ----------
    y_proba : np.ndarray
        Predicted churn probabilities.
    segment_definitions : Dict, optional
        Custom segment definitions.

    Returns
    -------
    np.ndarray
        Segment labels for each customer.
    """
    if segment_definitions is None:
        segment_definitions = SEGMENT_DEFINITIONS

    segments = np.empty(len(y_proba), dtype='U20')

    for seg_key, seg_def in segment_definitions.items():
        mask = (y_proba >= seg_def['threshold_low']) & (y_proba < seg_def['threshold_high'])
        segments[mask] = seg_def['name']

    return segments


def profile_segments(
    df: pd.DataFrame,
    segments: np.ndarray,
    y_proba: np.ndarray,
    feature_cols: List[str] = None
) -> Dict[str, Dict]:
    """
    Create detailed profiles for each segment.

    Parameters
    ----------
    df : pd.DataFrame
        Original customer data.
    segments : np.ndarray
        Segment labels.
    y_proba : np.ndarray
        Predicted churn probabilities.
    feature_cols : list, optional
        Features to include in profile.

    Returns
    -------
    Dict
        Profile for each segment.
    """
    df_analysis = df.copy()
    df_analysis['segment'] = segments
    df_analysis['churn_probability'] = y_proba

    profiles = {}

    for segment_name in df_analysis['segment'].unique():
        segment_df = df_analysis[df_analysis['segment'] == segment_name]

        profile = {
            'count': len(segment_df),
            'percentage': len(segment_df) / len(df_analysis) * 100,
            'avg_churn_probability': segment_df['churn_probability'].mean(),
            'probability_range': (
                segment_df['churn_probability'].min(),
                segment_df['churn_probability'].max()
            )
        }

        # Add feature statistics for key features
        key_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract']

        for feat in key_features:
            if feat in segment_df.columns:
                if segment_df[feat].dtype in ['int64', 'float64']:
                    profile[f'{feat}_mean'] = segment_df[feat].mean()
                    profile[f'{feat}_median'] = segment_df[feat].median()
                else:
                    profile[f'{feat}_distribution'] = segment_df[feat].value_counts(normalize=True).to_dict()

        profiles[segment_name] = profile

    return profiles


def get_segment_summary(profiles: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create a summary table of all segments.

    Returns
    -------
    pd.DataFrame
        Segment summary table.
    """
    summary_data = []

    for seg_name, seg_def in SEGMENT_DEFINITIONS.items():
        profile = profiles.get(seg_def['name'], {})

        summary_data.append({
            'Segment': seg_def['name'],
            'Count': profile.get('count', 0),
            'Percentage': f"{profile.get('percentage', 0):.1f}%",
            'Avg Churn Prob': f"{profile.get('avg_churn_probability', 0):.1%}",
            'Priority': seg_def['intervention_priority'],
            'Action': seg_def['recommended_action']
        })

    return pd.DataFrame(summary_data)


def get_top_risk_customers(
    df: pd.DataFrame,
    y_proba: np.ndarray,
    customer_id_col: str = None,
    top_n: int = 100
) -> pd.DataFrame:
    """
    Get the top N highest-risk customers.

    Returns
    -------
    pd.DataFrame
        Top risk customers with their details.
    """
    df_result = df.copy()
    df_result['churn_probability'] = y_proba
    df_result['risk_rank'] = df_result['churn_probability'].rank(ascending=False, method='first')
    df_result['segment'] = create_risk_segments(y_proba)

    # Sort by probability descending
    df_result = df_result.sort_values('churn_probability', ascending=False)

    return df_result.head(top_n)


def calculate_segment_roi(
    profiles: Dict[str, Dict],
    costs: Dict = None
) -> Dict[str, Dict]:
    """
    Calculate expected ROI for intervening on each segment.

    Parameters
    ----------
    profiles : Dict
        Segment profiles.
    costs : Dict, optional
        Business costs.

    Returns
    -------
    Dict
        ROI analysis for each segment.
    """
    if costs is None:
        costs = {
            'cost_of_churn': 500,
            'cost_of_retention': 100,
            'expected_retention_rate': 0.5  # 50% of at-risk customers retained
        }

    roi_analysis = {}

    for seg_name, profile in profiles.items():
        count = profile.get('count', 0)
        avg_prob = profile.get('avg_churn_probability', 0)

        # Expected churners in segment
        expected_churners = int(count * avg_prob)

        # Cost without intervention
        cost_no_action = expected_churners * costs['cost_of_churn']

        # Cost with intervention
        retention_cost = count * costs['cost_of_retention']
        retained_customers = int(expected_churners * costs['expected_retention_rate'])
        churners_after_intervention = expected_churners - retained_customers
        cost_with_action = retention_cost + (churners_after_intervention * costs['cost_of_churn'])

        # ROI
        savings = cost_no_action - cost_with_action
        roi = (savings / retention_cost * 100) if retention_cost > 0 else 0

        roi_analysis[seg_name] = {
            'customer_count': count,
            'expected_churners': expected_churners,
            'cost_no_action': cost_no_action,
            'cost_with_action': cost_with_action,
            'retention_investment': retention_cost,
            'expected_retained': retained_customers,
            'net_savings': savings,
            'roi_percent': roi,
            'recommend_intervention': savings > 0
        }

    return roi_analysis


def get_intervention_recommendations(
    roi_analysis: Dict[str, Dict],
    budget: float = None
) -> List[Dict]:
    """
    Generate prioritized intervention recommendations.

    Parameters
    ----------
    roi_analysis : Dict
        ROI analysis for each segment.
    budget : float, optional
        Available budget constraint.

    Returns
    -------
    List[Dict]
        Prioritized recommendations.
    """
    recommendations = []

    # Sort segments by ROI (descending)
    sorted_segments = sorted(
        roi_analysis.items(),
        key=lambda x: x[1]['roi_percent'],
        reverse=True
    )

    remaining_budget = budget
    total_savings = 0

    for seg_name, analysis in sorted_segments:
        if not analysis['recommend_intervention']:
            continue

        invest = analysis['retention_investment']
        savings = analysis['net_savings']

        if budget is not None:
            if invest > remaining_budget:
                continue
            remaining_budget -= invest

        total_savings += savings

        seg_def = next(
            (d for d in SEGMENT_DEFINITIONS.values() if d['name'] == seg_name),
            {}
        )

        recommendations.append({
            'segment': seg_name,
            'priority': seg_def.get('intervention_priority', 0),
            'customer_count': analysis['customer_count'],
            'investment_required': invest,
            'expected_savings': savings,
            'roi_percent': analysis['roi_percent'],
            'action': seg_def.get('recommended_action', 'Contact customer')
        })

    return recommendations


def generate_segment_report(
    profiles: Dict,
    roi_analysis: Dict
) -> str:
    """
    Generate a markdown segment analysis report.

    Returns
    -------
    str
        Formatted report.
    """
    report = "# Customer Segment Analysis\n\n"

    report += "## Segment Overview\n\n"
    report += "| Segment | Customers | Avg Churn Risk | Investment | Expected Savings | ROI |\n"
    report += "|---------|-----------|----------------|------------|------------------|-----|\n"

    for seg_name in ['Critical', 'At Risk', 'Monitor', 'Safe']:
        profile = profiles.get(seg_name, {})
        roi = roi_analysis.get(seg_name, {})

        report += f"| {seg_name} | {profile.get('count', 0):,} | "
        report += f"{profile.get('avg_churn_probability', 0):.1%} | "
        report += f"${roi.get('retention_investment', 0):,.0f} | "
        report += f"${roi.get('net_savings', 0):,.0f} | "
        report += f"{roi.get('roi_percent', 0):.0f}% |\n"

    report += "\n## Recommendations\n\n"

    for seg_def in sorted(SEGMENT_DEFINITIONS.values(), key=lambda x: x['intervention_priority']):
        seg_name = seg_def['name']
        profile = profiles.get(seg_name, {})

        report += f"### {seg_name} Segment\n\n"
        report += f"- **Size:** {profile.get('count', 0):,} customers "
        report += f"({profile.get('percentage', 0):.1f}%)\n"
        report += f"- **Action:** {seg_def['recommended_action']}\n"
        report += f"- **Description:** {seg_def['description']}\n\n"

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

    # Create segments
    segments = create_risk_segments(y_proba)

    # Profile segments
    profiles = profile_segments(test_df, segments, y_proba)

    print("\n=== Segment Summary ===")
    print(get_segment_summary(profiles))

    # ROI analysis
    roi = calculate_segment_roi(profiles)
    print("\n=== ROI Analysis ===")
    for seg, analysis in roi.items():
        if analysis['recommend_intervention']:
            print(f"{seg}: ROI = {analysis['roi_percent']:.0f}%, "
                 f"Savings = ${analysis['net_savings']:,.0f}")

    # Get recommendations
    recommendations = get_intervention_recommendations(roi)
    print("\n=== Intervention Recommendations ===")
    for rec in recommendations:
        print(f"Priority {rec['priority']}: {rec['segment']} - "
             f"Invest ${rec['investment_required']:,.0f} for "
             f"${rec['expected_savings']:,.0f} savings")
