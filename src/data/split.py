"""
Data Splitting Module
Creates stratified train/validation/test splits preserving churn ratio.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_stratified_splits(
    df: pd.DataFrame,
    target_col: str = 'Churn',
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    save_dir: Optional[str] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/validation/test splits.

    The splits preserve the churn ratio across all sets to ensure
    fair evaluation and prevent data leakage from imbalanced sampling.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with target column.
    target_col : str
        Name of target column for stratification.
    train_size : float
        Proportion for training set (default 0.70).
    val_size : float
        Proportion for validation set (default 0.15).
    test_size : float
        Proportion for test set (default 0.15).
    random_state : int
        Random seed for reproducibility.
    save_dir : str, optional
        Directory to save split CSVs. If None, doesn't save.
    verbose : bool
        If True, log split statistics.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df)
    """
    # Validate proportions
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split proportions must sum to 1.0, got {total}")

    if verbose:
        logger.info(f"Creating stratified splits: {train_size:.0%}/{val_size:.0%}/{test_size:.0%}")

    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        stratify=df[target_col],
        random_state=random_state
    )

    # Second split: val vs test (from temp)
    # Adjust proportions for the remaining data
    val_proportion = val_size / (val_size + test_size)

    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_proportion,
        stratify=temp_df[target_col],
        random_state=random_state
    )

    # Log statistics
    if verbose:
        _log_split_statistics(train_df, val_df, test_df, target_col)

    # Save to disk if requested
    if save_dir:
        _save_splits(train_df, val_df, test_df, save_dir, verbose)

    return train_df, val_df, test_df


def _log_split_statistics(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str
) -> None:
    """Log statistics about the splits."""
    total = len(train_df) + len(val_df) + len(test_df)

    logger.info("\n=== Split Statistics ===")
    logger.info(f"Total samples: {total}")

    for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        size = len(df)
        pct = size / total * 100
        churn_rate = df[target_col].mean() * 100
        logger.info(f"  {name}: {size} samples ({pct:.1f}%), Churn rate: {churn_rate:.2f}%")


def _save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    save_dir: str,
    verbose: bool
) -> None:
    """Save splits to CSV files."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(save_path / 'train.csv', index=False)
    val_df.to_csv(save_path / 'val.csv', index=False)
    test_df.to_csv(save_path / 'test.csv', index=False)

    if verbose:
        logger.info(f"Saved splits to {save_path}")


def load_splits(
    data_dir: str,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load previously saved splits from disk.

    Parameters
    ----------
    data_dir : str
        Directory containing train.csv, val.csv, test.csv
    verbose : bool
        If True, log loading statistics.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df)
    """
    data_path = Path(data_dir)

    train_df = pd.read_csv(data_path / 'train.csv')
    val_df = pd.read_csv(data_path / 'val.csv')
    test_df = pd.read_csv(data_path / 'test.csv')

    if verbose:
        logger.info(f"Loaded splits from {data_path}")
        logger.info(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    return train_df, val_df, test_df


def get_split_report(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = 'Churn'
) -> dict:
    """
    Generate a detailed report about the splits.

    Returns
    -------
    dict
        Detailed statistics about each split.
    """
    total = len(train_df) + len(val_df) + len(test_df)

    report = {
        'total_samples': total,
        'splits': {}
    }

    for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        report['splits'][name] = {
            'n_samples': len(df),
            'percentage': len(df) / total * 100,
            'churn_count': int(df[target_col].sum()),
            'no_churn_count': int((df[target_col] == 0).sum()),
            'churn_rate': df[target_col].mean() * 100
        }

    # Check stratification quality
    churn_rates = [report['splits'][s]['churn_rate'] for s in ['train', 'val', 'test']]
    report['stratification_variance'] = np.var(churn_rates)
    report['stratification_quality'] = 'Good' if np.var(churn_rates) < 0.5 else 'Check stratification'

    return report


if __name__ == "__main__":
    from load_data import load_raw_data
    from preprocess import preprocess_data
    from feature_engineering import engineer_features

    # Test full pipeline
    df_raw = load_raw_data()
    df_processed = preprocess_data(df_raw)
    df_engineered = engineer_features(df_processed)

    # Create and save splits
    project_root = Path(__file__).parent.parent.parent
    save_dir = project_root / 'data' / 'processed'

    train_df, val_df, test_df = create_stratified_splits(
        df_engineered,
        save_dir=str(save_dir)
    )

    # Generate report
    report = get_split_report(train_df, val_df, test_df)

    print("\n=== Split Report ===")
    print(f"Total samples: {report['total_samples']}")
    print(f"Stratification quality: {report['stratification_quality']}")

    for split_name, stats in report['splits'].items():
        print(f"\n{split_name.upper()}:")
        print(f"  Samples: {stats['n_samples']} ({stats['percentage']:.1f}%)")
        print(f"  Churn rate: {stats['churn_rate']:.2f}%")
