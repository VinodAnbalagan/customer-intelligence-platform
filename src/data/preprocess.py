"""
Data Preprocessing Module
Handles cleaning, type conversion, and initial transformations of the Telco Churn dataset.
"""

import pandas as pd
import numpy as np
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Comprehensive preprocessing of the Telco Churn dataset.

    Steps performed:
    1. Fix TotalCharges (convert string to float, handle blanks)
    2. Encode target variable (Churn: Yes=1, No=0)
    3. Drop customerID (not a predictive feature)
    4. Set proper dtypes for all columns

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe from load_data.
    verbose : bool
        If True, log preprocessing steps.

    Returns
    -------
    pd.DataFrame
        Cleaned and preprocessed dataframe.
    """
    df = df.copy()

    if verbose:
        logger.info("Starting preprocessing...")

    # Step 1: Fix TotalCharges
    # TotalCharges has 11 rows with " " (space) - these are new customers with tenure=0
    df = _fix_total_charges(df, verbose)

    # Step 2: Encode target variable
    df = _encode_target(df, verbose)

    # Step 3: Drop customerID
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        if verbose:
            logger.info("Dropped customerID column")

    # Step 4: Set proper dtypes
    df = _set_dtypes(df, verbose)

    if verbose:
        logger.info(f"Preprocessing complete. Shape: {df.shape}")

    return df


def _fix_total_charges(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Fix TotalCharges column: convert to numeric and handle missing/blank values.

    The dataset has 11 rows where TotalCharges is " " (a space string).
    These are new customers with tenure=0, so TotalCharges should be 0 or
    equal to MonthlyCharges (first month).
    """
    df = df.copy()

    # Count problematic rows before fix
    blank_count = (df['TotalCharges'] == ' ').sum() if df['TotalCharges'].dtype == 'object' else 0

    if verbose and blank_count > 0:
        logger.info(f"Found {blank_count} rows with blank TotalCharges (new customers)")

    # Convert to numeric, coercing errors to NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Fill NaN values with MonthlyCharges (these are new customers, first month)
    # This is more realistic than filling with 0
    mask = df['TotalCharges'].isna()
    df.loc[mask, 'TotalCharges'] = df.loc[mask, 'MonthlyCharges']

    if verbose:
        logger.info(f"Converted TotalCharges to numeric. "
                   f"Filled {mask.sum()} missing values with MonthlyCharges")

    return df


def _encode_target(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Encode the target variable Churn: Yes=1, No=0.
    """
    df = df.copy()

    if df['Churn'].dtype == 'object':
        original_values = df['Churn'].unique()
        df['Churn'] = (df['Churn'] == 'Yes').astype(int)

        if verbose:
            logger.info(f"Encoded Churn: {list(original_values)} -> [0, 1]")

    return df


def _set_dtypes(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Set appropriate dtypes for all columns.

    Notes:
    - SeniorCitizen is already 0/1 in the raw data (not Yes/No)
    - Numerical: tenure, MonthlyCharges, TotalCharges, SeniorCitizen
    - Categorical: all others except target
    """
    df = df.copy()

    # Numerical columns (already numeric or should be)
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

    for col in numerical_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # SeniorCitizen is already 0/1
    if 'SeniorCitizen' in df.columns:
        df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)

    # Categorical columns - convert to category dtype for efficiency
    categorical_cols = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod'
    ]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    if verbose:
        logger.info(f"Set dtypes: {len(numerical_cols)} numerical, "
                   f"{len(categorical_cols)} categorical")

    return df


def get_preprocessing_report(df_raw: pd.DataFrame, df_processed: pd.DataFrame) -> dict:
    """
    Generate a report comparing raw and processed data.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Original raw dataframe.
    df_processed : pd.DataFrame
        Preprocessed dataframe.

    Returns
    -------
    dict
        Report with before/after comparisons.
    """
    report = {
        'raw_shape': df_raw.shape,
        'processed_shape': df_processed.shape,
        'columns_removed': list(set(df_raw.columns) - set(df_processed.columns)),
        'columns_added': list(set(df_processed.columns) - set(df_raw.columns)),
        'missing_values_before': df_raw.isnull().sum().sum(),
        'missing_values_after': df_processed.isnull().sum().sum(),
        'dtypes_after': df_processed.dtypes.astype(str).to_dict(),
        'churn_distribution': df_processed['Churn'].value_counts().to_dict(),
        'churn_rate': df_processed['Churn'].mean()
    }

    return report


if __name__ == "__main__":
    from load_data import load_raw_data

    # Test preprocessing
    df_raw = load_raw_data()
    df_processed = preprocess_data(df_raw)

    report = get_preprocessing_report(df_raw, df_processed)

    print("\n=== Preprocessing Report ===")
    print(f"Raw shape: {report['raw_shape']}")
    print(f"Processed shape: {report['processed_shape']}")
    print(f"Columns removed: {report['columns_removed']}")
    print(f"Missing values: {report['missing_values_before']} -> {report['missing_values_after']}")
    print(f"Churn rate: {report['churn_rate']:.2%}")
