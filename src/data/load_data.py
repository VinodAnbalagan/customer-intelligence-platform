"""
Data Loading Module
Handles loading and validation of the Telco Customer Churn dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Expected schema for validation
EXPECTED_COLUMNS = [
    'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
    'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
]

EXPECTED_ROW_COUNT = 7043


def load_raw_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load the Telco Customer Churn dataset from CSV.

    Parameters
    ----------
    filepath : str, optional
        Path to the CSV file. If None, uses default location.

    Returns
    -------
    pd.DataFrame
        Raw dataframe loaded from CSV.
    """
    if filepath is None:
        # Default path relative to project root
        project_root = Path(__file__).parent.parent.parent
        filepath = project_root / 'data' / 'raw' / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. "
            f"Please download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn "
            f"and place the CSV in data/raw/"
        )

    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")

    return df


def validate_data(df: pd.DataFrame) -> Tuple[bool, list]:
    """
    Validate the loaded dataframe against expected schema.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded dataframe to validate.

    Returns
    -------
    Tuple[bool, list]
        (is_valid, list of validation issues)
    """
    issues = []

    # Check row count
    if len(df) != EXPECTED_ROW_COUNT:
        issues.append(f"Expected {EXPECTED_ROW_COUNT} rows, got {len(df)}")

    # Check columns
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    extra_cols = set(df.columns) - set(EXPECTED_COLUMNS)

    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    if extra_cols:
        issues.append(f"Unexpected columns: {extra_cols}")

    # Check for critical columns
    if 'Churn' not in df.columns:
        issues.append("Target column 'Churn' is missing!")

    if 'customerID' not in df.columns:
        issues.append("Customer identifier 'customerID' is missing!")

    is_valid = len(issues) == 0

    if is_valid:
        logger.info("Data validation passed!")
    else:
        logger.warning(f"Data validation found {len(issues)} issues:")
        for issue in issues:
            logger.warning(f"  - {issue}")

    return is_valid, issues


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Generate a summary of the dataset for quick inspection.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to summarize.

    Returns
    -------
    dict
        Summary statistics and information.
    """
    summary = {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'columns': list(df.columns),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
    }

    # Target distribution if available
    if 'Churn' in df.columns:
        churn_counts = df['Churn'].value_counts()
        summary['churn_distribution'] = churn_counts.to_dict()
        summary['churn_rate'] = (df['Churn'] == 'Yes').mean() if df['Churn'].dtype == 'object' else df['Churn'].mean()

    return summary


if __name__ == "__main__":
    # Quick test
    df = load_raw_data()
    is_valid, issues = validate_data(df)
    summary = get_data_summary(df)

    print("\n=== Data Summary ===")
    print(f"Rows: {summary['n_rows']}")
    print(f"Columns: {summary['n_columns']}")
    print(f"Churn Rate: {summary.get('churn_rate', 'N/A'):.2%}")
    print(f"Memory: {summary['memory_usage_mb']:.2f} MB")
