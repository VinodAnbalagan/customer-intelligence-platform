"""
Feature Engineering Module
Creates meaningful business features from the Telco Churn dataset.

Each feature is designed with a specific business rationale documented below.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Service columns used for counting services
SERVICE_COLUMNS = [
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies'
]


def engineer_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Create all engineered features for the churn prediction model.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataframe from preprocess.py
    verbose : bool
        If True, log feature creation steps.

    Returns
    -------
    pd.DataFrame
        Dataframe with all engineered features added.
    """
    df = df.copy()

    if verbose:
        logger.info("Starting feature engineering...")
        initial_cols = len(df.columns)

    # Tenure-based features
    df = _create_tenure_features(df)

    # Charge-based features
    df = _create_charge_features(df)

    # Service-based features
    df = _create_service_features(df)

    # Contract and payment features
    df = _create_contract_features(df)

    # Customer profile features
    df = _create_profile_features(df)

    if verbose:
        new_cols = len(df.columns) - initial_cols
        logger.info(f"Feature engineering complete. Added {new_cols} new features.")

    return df


def _create_tenure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create tenure-related features.

    Business Rationale:
    - tenure_group: Customer lifecycle stages have different churn risks.
      New customers (0-12 months) are highest risk, long-term customers (49+ months)
      are most stable. Grouping allows the model to capture these lifecycle patterns.
    - is_new_customer: First 6 months are critical for retention. New customers
      haven't developed loyalty yet and are evaluating the service.
    """
    df = df.copy()

    # Tenure groups: lifecycle stages
    bins = [0, 12, 24, 48, 72, np.inf]
    labels = ['0-12', '13-24', '25-48', '49-72', '72+']
    df['tenure_group'] = pd.cut(df['tenure'], bins=bins, labels=labels, include_lowest=True)

    # New customer flag (first 6 months - critical retention period)
    df['is_new_customer'] = (df['tenure'] <= 6).astype(int)

    return df


def _create_charge_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create charge-related features.

    Business Rationale:
    - monthly_to_total_ratio: High ratio indicates new customers or billing anomalies.
      A stable customer should have low ratio (many months of charges).
    - avg_monthly_charge: Actual average spend per month. Differs from MonthlyCharges
      if customer had plan changes. Inconsistency might indicate dissatisfaction.
    - charge_consistency: How close current rate is to historical average.
      Large deviations might indicate recent price changes (churn trigger).
    - high_monthly_charge: Above-median monthly charge. Price-sensitive customers
      with high bills may be more likely to churn.
    """
    df = df.copy()

    # Monthly to total ratio (high = new customer)
    # Add 1 to avoid division by zero
    df['monthly_to_total_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)

    # Average monthly charge based on history
    # Add 1 to tenure to avoid division by zero for new customers
    df['avg_monthly_charge'] = df['TotalCharges'] / (df['tenure'] + 1)

    # Charge consistency: ratio of current to average
    # Values close to 1 = consistent pricing, far from 1 = recent changes
    df['charge_consistency'] = np.where(
        df['avg_monthly_charge'] > 0,
        df['MonthlyCharges'] / df['avg_monthly_charge'],
        1.0
    )

    # High monthly charge flag (above median)
    median_charge = df['MonthlyCharges'].median()
    df['high_monthly_charge'] = (df['MonthlyCharges'] > median_charge).astype(int)

    return df


def _create_service_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create service-related features.

    Business Rationale:
    - service_count: More services = more invested in the platform = lower churn.
      Customers with multiple services have higher switching costs.
    - has_premium_support: OnlineSecurity or TechSupport indicates customer values
      support. These customers may have different expectations and behaviors.
    - has_streaming: StreamingTV or StreamingMovies indicates entertainment use.
      These customers use the internet service for specific value-add services.
    """
    df = df.copy()

    # Count of "Yes" services
    def count_services(row):
        count = 0
        for col in SERVICE_COLUMNS:
            if col in row.index:
                val = str(row[col])
                if val == 'Yes' or (col == 'InternetService' and val != 'No'):
                    count += 1
        return count

    df['service_count'] = df.apply(count_services, axis=1)

    # Premium support flag
    df['has_premium_support'] = (
        (df['OnlineSecurity'].astype(str) == 'Yes') |
        (df['TechSupport'].astype(str) == 'Yes')
    ).astype(int)

    # Streaming services flag
    df['has_streaming'] = (
        (df['StreamingTV'].astype(str) == 'Yes') |
        (df['StreamingMovies'].astype(str) == 'Yes')
    ).astype(int)

    return df


def _create_contract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create contract and payment features.

    Business Rationale:
    - contract_value: Numerical encoding of contract commitment.
      Month-to-month (0) = no commitment = highest churn risk.
      Two year (2) = high commitment = lowest churn risk.
    - auto_payment: Automatic payments reduce friction and indicate trust.
      Customers with auto-pay are less likely to actively cancel.
    """
    df = df.copy()

    # Contract value mapping (commitment level)
    contract_map = {
        'Month-to-month': 0,
        'One year': 1,
        'Two year': 2
    }
    df['contract_value'] = df['Contract'].astype(str).map(contract_map).fillna(0).astype(int)

    # Auto payment flag
    df['auto_payment'] = df['PaymentMethod'].astype(str).str.contains(
        'automatic', case=False
    ).astype(int)

    return df


def _create_profile_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create customer profile features.

    Business Rationale:
    - has_dependents_or_partner: Customers with family ties may have different
      service needs and stability. Family accounts often have lower churn.
    - is_senior_alone: Senior citizens without partner/dependents may need
      different support and have different churn patterns.
    """
    df = df.copy()

    # Family connection flag
    df['has_family'] = (
        (df['Partner'].astype(str) == 'Yes') |
        (df['Dependents'].astype(str) == 'Yes')
    ).astype(int)

    # Senior living alone (potentially vulnerable segment)
    df['is_senior_alone'] = (
        (df['SeniorCitizen'] == 1) &
        (df['Partner'].astype(str) == 'No') &
        (df['Dependents'].astype(str) == 'No')
    ).astype(int)

    return df


def get_feature_documentation() -> dict:
    """
    Return documentation for all engineered features.

    Returns
    -------
    dict
        Feature names mapped to their business rationale.
    """
    return {
        'tenure_group': (
            "Customer lifecycle stage (0-12, 13-24, 25-48, 49-72, 72+ months). "
            "Different stages have different churn risks and needs."
        ),
        'is_new_customer': (
            "Binary flag for customers in first 6 months. "
            "Critical retention period before loyalty develops."
        ),
        'monthly_to_total_ratio': (
            "Ratio of monthly charges to total charges. "
            "High values indicate new customers or billing anomalies."
        ),
        'avg_monthly_charge': (
            "Historical average monthly spend (TotalCharges / tenure). "
            "Differs from current rate if customer had plan changes."
        ),
        'charge_consistency': (
            "Ratio of current to average monthly charge. "
            "Values far from 1.0 indicate recent pricing changes."
        ),
        'high_monthly_charge': (
            "Binary flag for above-median monthly charges. "
            "Price-sensitive customers may churn more."
        ),
        'service_count': (
            "Count of subscribed services. "
            "More services = higher switching costs = lower churn."
        ),
        'has_premium_support': (
            "Has OnlineSecurity or TechSupport. "
            "Indicates customer values support services."
        ),
        'has_streaming': (
            "Has StreamingTV or StreamingMovies. "
            "Entertainment users with specific value-add needs."
        ),
        'contract_value': (
            "Numerical contract commitment (0=month-to-month, 1=1yr, 2=2yr). "
            "Higher commitment = lower churn risk."
        ),
        'auto_payment': (
            "Uses automatic payment method. "
            "Auto-pay reduces friction and indicates trust."
        ),
        'has_family': (
            "Has partner or dependents. "
            "Family accounts often have different needs and stability."
        ),
        'is_senior_alone': (
            "Senior citizen without partner or dependents. "
            "Potentially vulnerable segment needing different approach."
        )
    }


def get_engineered_feature_names() -> List[str]:
    """Return list of all engineered feature names."""
    return [
        'tenure_group', 'is_new_customer',
        'monthly_to_total_ratio', 'avg_monthly_charge',
        'charge_consistency', 'high_monthly_charge',
        'service_count', 'has_premium_support', 'has_streaming',
        'contract_value', 'auto_payment',
        'has_family', 'is_senior_alone'
    ]


if __name__ == "__main__":
    from load_data import load_raw_data
    from preprocess import preprocess_data

    # Test feature engineering
    df_raw = load_raw_data()
    df_processed = preprocess_data(df_raw)
    df_engineered = engineer_features(df_processed)

    print("\n=== Feature Engineering Report ===")
    print(f"Original features: {len(df_processed.columns)}")
    print(f"Total features: {len(df_engineered.columns)}")
    print(f"New features: {len(df_engineered.columns) - len(df_processed.columns)}")

    print("\n=== Engineered Features ===")
    for feat in get_engineered_feature_names():
        if feat in df_engineered.columns:
            print(f"  {feat}: {df_engineered[feat].dtype}")
