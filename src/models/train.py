"""
Model Training Module
Trains and compares multiple classification models for churn prediction.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss,
    confusion_matrix, classification_report
)
import xgboost as xgb
import lightgbm as lgb
import joblib
from pathlib import Path
from typing import Dict, Tuple, Any, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Categorical columns that need encoding
CATEGORICAL_COLS = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod', 'tenure_group'
]


def prepare_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = 'Churn'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Prepare features for modeling: encode categoricals, scale numericals.

    Returns
    -------
    Tuple containing:
        X_train, X_val, X_test, y_train, y_val, y_test, preprocessors_dict
    """
    # Separate features and target
    feature_cols = [c for c in train_df.columns if c != target_col]

    X_train = train_df[feature_cols].copy()
    X_val = val_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    y_test = test_df[target_col].values

    # Identify column types
    cat_cols = [c for c in CATEGORICAL_COLS if c in X_train.columns]
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    # Encode categorical columns
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        # Fit on combined data to handle all possible values
        all_values = pd.concat([X_train[col], X_val[col], X_test[col]]).astype(str)
        le.fit(all_values)

        X_train[col] = le.transform(X_train[col].astype(str))
        X_val[col] = le.transform(X_val[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

        label_encoders[col] = le

    # Scale numerical columns
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_val[num_cols] = scaler.transform(X_val[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    preprocessors = {
        'label_encoders': label_encoders,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'cat_cols': cat_cols,
        'num_cols': num_cols
    }

    return (
        X_train.values, X_val.values, X_test.values,
        y_train, y_val, y_test,
        preprocessors
    )


def get_models(class_weight_ratio: float = None) -> Dict[str, Any]:
    """
    Get dictionary of models to train.

    Parameters
    ----------
    class_weight_ratio : float, optional
        Ratio of negative to positive class for handling imbalance.

    Returns
    -------
    Dict[str, model]
        Dictionary of model name to model instance.
    """
    models = {
        'logistic_regression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'xgboost': xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=class_weight_ratio if class_weight_ratio else 1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        ),
        'lightgbm': lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            is_unbalance=True,
            random_state=42,
            verbose=-1
        )
    }
    return models


def evaluate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    model_name: str = ''
) -> Dict[str, float]:
    """
    Evaluate a model on given data.

    Parameters
    ----------
    model : trained model
        Model with predict and predict_proba methods.
    X : np.ndarray
        Features.
    y : np.ndarray
        True labels.
    model_name : str
        Name for logging.

    Returns
    -------
    Dict[str, float]
        Dictionary of metrics.
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'f1_macro': f1_score(y, y_pred, average='macro'),
        'roc_auc': roc_auc_score(y, y_proba),
        'pr_auc': average_precision_score(y, y_proba),
        'log_loss': log_loss(y, y_proba)
    }

    return metrics


def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    verbose: bool = True
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
    """
    Train all models and evaluate on validation set.

    Returns
    -------
    Tuple[Dict[str, model], Dict[str, Dict[str, float]]]
        Trained models and their validation metrics.
    """
    # Calculate class weight ratio for XGBoost
    class_weight_ratio = (y_train == 0).sum() / (y_train == 1).sum()

    models = get_models(class_weight_ratio)
    trained_models = {}
    all_metrics = {}

    for name, model in models.items():
        if verbose:
            logger.info(f"Training {name}...")

        model.fit(X_train, y_train)
        trained_models[name] = model

        metrics = evaluate_model(model, X_val, y_val, name)
        all_metrics[name] = metrics

        if verbose:
            logger.info(f"  {name} - F1: {metrics['f1']:.4f}, "
                       f"AUC-ROC: {metrics['roc_auc']:.4f}, "
                       f"AUC-PR: {metrics['pr_auc']:.4f}")

    return trained_models, all_metrics


def get_feature_importance(
    model: Any,
    feature_names: List[str],
    model_type: str
) -> pd.DataFrame:
    """
    Extract feature importance from a trained model.

    Returns
    -------
    pd.DataFrame
        Feature importance sorted by importance.
    """
    if model_type == 'logistic_regression':
        importance = np.abs(model.coef_[0])
    elif model_type in ['random_forest', 'xgboost', 'lightgbm']:
        importance = model.feature_importances_
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })

    return df.sort_values('importance', ascending=False).reset_index(drop=True)


def get_model_comparison_table(all_metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Create a comparison table of all models.

    Returns
    -------
    pd.DataFrame
        Comparison table with models as rows and metrics as columns.
    """
    df = pd.DataFrame(all_metrics).T
    df = df.round(4)
    df = df.sort_values('f1', ascending=False)
    return df


def save_model(
    model: Any,
    preprocessors: Dict,
    model_path: str,
    preprocessor_path: str
) -> None:
    """Save model and preprocessors to disk."""
    joblib.dump(model, model_path)
    joblib.dump(preprocessors, preprocessor_path)
    logger.info(f"Saved model to {model_path}")
    logger.info(f"Saved preprocessors to {preprocessor_path}")


def load_model(
    model_path: str,
    preprocessor_path: str
) -> Tuple[Any, Dict]:
    """Load model and preprocessors from disk."""
    model = joblib.load(model_path)
    preprocessors = joblib.load(preprocessor_path)
    return model, preprocessors


if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent))

    from data.load_data import load_raw_data
    from data.preprocess import preprocess_data
    from data.feature_engineering import engineer_features
    from data.split import create_stratified_splits

    # Load and prepare data
    logger.info("Loading and preparing data...")
    df = load_raw_data()
    df = preprocess_data(df)
    df = engineer_features(df)

    train_df, val_df, test_df = create_stratified_splits(df)

    # Prepare features
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessors = prepare_features(
        train_df, val_df, test_df
    )

    # Train models
    trained_models, val_metrics = train_models(X_train, y_train, X_val, y_val)

    # Print comparison
    print("\n=== Model Comparison (Validation Set) ===")
    comparison = get_model_comparison_table(val_metrics)
    print(comparison)

    # Get best model
    best_model_name = comparison.index[0]
    best_model = trained_models[best_model_name]

    # Evaluate on test set
    print(f"\n=== Best Model ({best_model_name}) on Test Set ===")
    test_metrics = evaluate_model(best_model, X_test, y_test, best_model_name)
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Feature importance
    print(f"\n=== Feature Importance ({best_model_name}) ===")
    importance = get_feature_importance(
        best_model,
        preprocessors['feature_cols'],
        best_model_name
    )
    print(importance.head(10))
