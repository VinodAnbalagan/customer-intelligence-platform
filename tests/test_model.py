"""
Tests for model training and prediction modules.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.train import (
    prepare_features, get_models, evaluate_model,
    train_models, get_feature_importance
)


@pytest.fixture
def sample_processed_data():
    """Create sample processed data for testing."""
    np.random.seed(42)
    n = 100

    data = {
        'gender': np.random.choice([0, 1], n),
        'SeniorCitizen': np.random.choice([0, 1], n),
        'Partner': np.random.choice([0, 1], n),
        'Dependents': np.random.choice([0, 1], n),
        'tenure': np.random.randint(1, 72, n),
        'PhoneService': np.random.choice([0, 1], n),
        'MultipleLines': np.random.choice([0, 1, 2], n),
        'InternetService': np.random.choice([0, 1, 2], n),
        'OnlineSecurity': np.random.choice([0, 1, 2], n),
        'OnlineBackup': np.random.choice([0, 1, 2], n),
        'DeviceProtection': np.random.choice([0, 1, 2], n),
        'TechSupport': np.random.choice([0, 1, 2], n),
        'StreamingTV': np.random.choice([0, 1, 2], n),
        'StreamingMovies': np.random.choice([0, 1, 2], n),
        'Contract': np.random.choice([0, 1, 2], n),
        'PaperlessBilling': np.random.choice([0, 1], n),
        'PaymentMethod': np.random.choice([0, 1, 2, 3], n),
        'MonthlyCharges': np.random.uniform(20, 120, n),
        'TotalCharges': np.random.uniform(100, 8000, n),
        'tenure_group': np.random.choice(['0-12', '13-24', '25-48', '49-72'], n),
        'is_new_customer': np.random.choice([0, 1], n),
        'service_count': np.random.randint(0, 9, n),
        'contract_value': np.random.choice([0, 1, 2], n),
        'Churn': np.random.choice([0, 1], n, p=[0.73, 0.27])  # ~27% churn
    }

    return pd.DataFrame(data)


@pytest.fixture
def train_val_test_data(sample_processed_data):
    """Split sample data into train/val/test."""
    n = len(sample_processed_data)
    train_idx = int(n * 0.7)
    val_idx = int(n * 0.85)

    train = sample_processed_data.iloc[:train_idx].copy()
    val = sample_processed_data.iloc[train_idx:val_idx].copy()
    test = sample_processed_data.iloc[val_idx:].copy()

    return train, val, test


class TestFeaturePreparation:
    """Tests for feature preparation."""

    def test_prepare_features_shapes(self, train_val_test_data):
        """Test that prepare_features returns correct shapes."""
        train, val, test = train_val_test_data

        X_train, X_val, X_test, y_train, y_val, y_test, prep = prepare_features(
            train, val, test
        )

        assert X_train.shape[0] == len(train)
        assert X_val.shape[0] == len(val)
        assert X_test.shape[0] == len(test)

        assert len(y_train) == len(train)
        assert len(y_val) == len(val)
        assert len(y_test) == len(test)

    def test_prepare_features_removes_target(self, train_val_test_data):
        """Test that target column is not in features."""
        train, val, test = train_val_test_data

        X_train, _, _, _, _, _, prep = prepare_features(train, val, test)

        assert 'Churn' not in prep['feature_cols']

    def test_preprocessors_returned(self, train_val_test_data):
        """Test that preprocessors dict has expected keys."""
        train, val, test = train_val_test_data

        _, _, _, _, _, _, prep = prepare_features(train, val, test)

        assert 'scaler' in prep
        assert 'feature_cols' in prep
        assert 'cat_cols' in prep
        assert 'num_cols' in prep


class TestModelTraining:
    """Tests for model training."""

    def test_get_models_returns_all_models(self):
        """Test that get_models returns all expected models."""
        models = get_models()

        expected = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm']
        for name in expected:
            assert name in models

    def test_train_models_returns_trained_models(self, train_val_test_data):
        """Test that train_models returns trained models."""
        train, val, test = train_val_test_data

        X_train, X_val, _, y_train, y_val, _, _ = prepare_features(train, val, test)

        trained_models, metrics = train_models(
            X_train, y_train, X_val, y_val, verbose=False
        )

        assert len(trained_models) > 0
        assert len(metrics) > 0

        for name, model in trained_models.items():
            assert hasattr(model, 'predict')
            assert hasattr(model, 'predict_proba')


class TestModelEvaluation:
    """Tests for model evaluation."""

    def test_evaluate_model_returns_all_metrics(self, train_val_test_data):
        """Test that evaluate_model returns all expected metrics."""
        train, val, test = train_val_test_data

        X_train, X_val, _, y_train, y_val, _, _ = prepare_features(train, val, test)

        models = get_models()
        model = models['logistic_regression']
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_val, y_val)

        expected_metrics = [
            'accuracy', 'precision', 'recall', 'f1',
            'f1_macro', 'roc_auc', 'pr_auc', 'log_loss'
        ]

        for metric in expected_metrics:
            assert metric in metrics

    def test_metrics_in_valid_range(self, train_val_test_data):
        """Test that metrics are in valid ranges."""
        train, val, test = train_val_test_data

        X_train, X_val, _, y_train, y_val, _, _ = prepare_features(train, val, test)

        models = get_models()
        model = models['logistic_regression']
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_val, y_val)

        # These should be between 0 and 1
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']:
            assert 0 <= metrics[metric] <= 1, f"{metric} out of range"

        # Log loss should be positive
        assert metrics['log_loss'] >= 0


class TestPredictions:
    """Tests for model predictions."""

    def test_predict_returns_binary(self, train_val_test_data):
        """Test that predict returns binary values."""
        train, val, test = train_val_test_data

        X_train, X_val, _, y_train, _, _, _ = prepare_features(train, val, test)

        models = get_models()
        model = models['xgboost']
        model.fit(X_train, y_train)

        predictions = model.predict(X_val)

        assert set(predictions).issubset({0, 1})

    def test_predict_proba_returns_valid_probabilities(self, train_val_test_data):
        """Test that predict_proba returns valid probabilities."""
        train, val, test = train_val_test_data

        X_train, X_val, _, y_train, _, _, _ = prepare_features(train, val, test)

        models = get_models()
        model = models['xgboost']
        model.fit(X_train, y_train)

        probas = model.predict_proba(X_val)

        # Should have 2 columns (negative and positive class)
        assert probas.shape[1] == 2

        # All probabilities should be between 0 and 1
        assert np.all(probas >= 0)
        assert np.all(probas <= 1)

        # Rows should sum to 1
        row_sums = probas.sum(axis=1)
        assert np.allclose(row_sums, 1.0)

    def test_prediction_shape_matches_input(self, train_val_test_data):
        """Test that prediction shape matches input."""
        train, val, test = train_val_test_data

        X_train, X_val, _, y_train, _, _, _ = prepare_features(train, val, test)

        models = get_models()
        model = models['random_forest']
        model.fit(X_train, y_train)

        predictions = model.predict(X_val)
        probas = model.predict_proba(X_val)

        assert len(predictions) == X_val.shape[0]
        assert probas.shape[0] == X_val.shape[0]


class TestFeatureImportance:
    """Tests for feature importance extraction."""

    def test_feature_importance_shape(self, train_val_test_data):
        """Test that feature importance has correct shape."""
        train, val, test = train_val_test_data

        X_train, _, _, y_train, _, _, prep = prepare_features(train, val, test)

        models = get_models()
        model = models['xgboost']
        model.fit(X_train, y_train)

        importance = get_feature_importance(
            model, prep['feature_cols'], 'xgboost'
        )

        assert len(importance) == len(prep['feature_cols'])
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns

    def test_feature_importance_sorted(self, train_val_test_data):
        """Test that feature importance is sorted descending."""
        train, val, test = train_val_test_data

        X_train, _, _, y_train, _, _, prep = prepare_features(train, val, test)

        models = get_models()
        model = models['random_forest']
        model.fit(X_train, y_train)

        importance = get_feature_importance(
            model, prep['feature_cols'], 'random_forest'
        )

        # Check that importance values are sorted descending
        values = importance['importance'].values
        assert all(values[i] >= values[i+1] for i in range(len(values)-1))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
