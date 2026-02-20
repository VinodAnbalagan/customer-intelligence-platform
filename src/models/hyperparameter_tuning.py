"""
Hyperparameter Tuning Module
Uses Optuna to optimize model hyperparameters.
"""

import numpy as np
import optuna
from optuna.samplers import TPESampler
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from typing import Dict, Any, Tuple, Optional
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


def create_xgboost_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weight_ratio: float
):
    """Create Optuna objective function for XGBoost."""

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'scale_pos_weight': class_weight_ratio,
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)

        y_pred = model.predict(X_val)
        return f1_score(y_val, y_pred)

    return objective


def create_lightgbm_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
):
    """Create Optuna objective function for LightGBM."""

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'is_unbalance': True,
            'random_state': 42,
            'verbose': -1
        }

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        return f1_score(y_val, y_pred)

    return objective


def create_rf_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
):
    """Create Optuna objective function for Random Forest."""

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        return f1_score(y_val, y_pred)

    return objective


def tune_model(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 100,
    verbose: bool = True
) -> Tuple[Dict[str, Any], Any]:
    """
    Tune hyperparameters for a given model type.

    Parameters
    ----------
    model_type : str
        One of 'xgboost', 'lightgbm', 'random_forest'
    X_train, y_train : training data
    X_val, y_val : validation data
    n_trials : int
        Number of Optuna trials.
    verbose : bool
        If True, log progress.

    Returns
    -------
    Tuple[Dict, model]
        Best parameters and trained model with those parameters.
    """
    if verbose:
        logger.info(f"Tuning {model_type} with {n_trials} trials...")

    # Calculate class weight ratio
    class_weight_ratio = (y_train == 0).sum() / (y_train == 1).sum()

    # Create objective function
    if model_type == 'xgboost':
        objective = create_xgboost_objective(
            X_train, y_train, X_val, y_val, class_weight_ratio
        )
    elif model_type == 'lightgbm':
        objective = create_lightgbm_objective(X_train, y_train, X_val, y_val)
    elif model_type == 'random_forest':
        objective = create_rf_objective(X_train, y_train, X_val, y_val)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Run optimization
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

    best_params = study.best_params

    if verbose:
        logger.info(f"Best F1 score: {study.best_value:.4f}")
        logger.info(f"Best params: {best_params}")

    # Train final model with best params
    if model_type == 'xgboost':
        best_params['scale_pos_weight'] = class_weight_ratio
        best_params['random_state'] = 42
        best_params['use_label_encoder'] = False
        best_params['eval_metric'] = 'logloss'
        best_model = xgb.XGBClassifier(**best_params)
    elif model_type == 'lightgbm':
        best_params['is_unbalance'] = True
        best_params['random_state'] = 42
        best_params['verbose'] = -1
        best_model = lgb.LGBMClassifier(**best_params)
    elif model_type == 'random_forest':
        best_params['class_weight'] = 'balanced'
        best_params['random_state'] = 42
        best_params['n_jobs'] = -1
        best_model = RandomForestClassifier(**best_params)

    best_model.fit(X_train, y_train)

    return best_params, best_model, study


def get_tuning_report(study: optuna.Study) -> Dict:
    """
    Generate a report from an Optuna study.

    Returns
    -------
    dict
        Summary of the tuning process.
    """
    return {
        'best_value': study.best_value,
        'best_params': study.best_params,
        'n_trials': len(study.trials),
        'best_trial_number': study.best_trial.number,
        'optimization_history': [
            {'trial': t.number, 'value': t.value}
            for t in study.trials if t.value is not None
        ]
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))

    from data.load_data import load_raw_data
    from data.preprocess import preprocess_data
    from data.feature_engineering import engineer_features
    from data.split import create_stratified_splits
    from train import prepare_features, evaluate_model

    # Load and prepare data
    df = load_raw_data()
    df = preprocess_data(df)
    df = engineer_features(df)
    train_df, val_df, test_df = create_stratified_splits(df)

    X_train, X_val, X_test, y_train, y_val, y_test, preprocessors = prepare_features(
        train_df, val_df, test_df
    )

    # Tune XGBoost (using fewer trials for demo)
    best_params, best_model, study = tune_model(
        'xgboost',
        X_train, y_train,
        X_val, y_val,
        n_trials=20  # Use 100 for production
    )

    # Evaluate tuned model
    print("\n=== Tuned XGBoost on Test Set ===")
    test_metrics = evaluate_model(best_model, X_test, y_test)
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
