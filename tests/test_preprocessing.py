"""
Tests for data preprocessing modules.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.preprocess import preprocess_data, _fix_total_charges, _encode_target
from data.feature_engineering import engineer_features, get_engineered_feature_names
from data.split import create_stratified_splits


@pytest.fixture
def sample_raw_data():
    """Create sample raw data for testing."""
    return pd.DataFrame({
        'customerID': ['001', '002', '003', '004', '005'],
        'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'SeniorCitizen': [0, 1, 0, 0, 1],
        'Partner': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'Dependents': ['No', 'No', 'Yes', 'No', 'Yes'],
        'tenure': [1, 34, 2, 45, 72],
        'PhoneService': ['Yes', 'Yes', 'No', 'Yes', 'Yes'],
        'MultipleLines': ['No', 'Yes', 'No phone service', 'Yes', 'Yes'],
        'InternetService': ['DSL', 'Fiber optic', 'No', 'Fiber optic', 'DSL'],
        'OnlineSecurity': ['No', 'Yes', 'No internet service', 'No', 'Yes'],
        'OnlineBackup': ['No', 'Yes', 'No internet service', 'No', 'Yes'],
        'DeviceProtection': ['No', 'Yes', 'No internet service', 'Yes', 'Yes'],
        'TechSupport': ['No', 'Yes', 'No internet service', 'No', 'Yes'],
        'StreamingTV': ['No', 'Yes', 'No internet service', 'Yes', 'Yes'],
        'StreamingMovies': ['No', 'Yes', 'No internet service', 'Yes', 'Yes'],
        'Contract': ['Month-to-month', 'Two year', 'Month-to-month', 'One year', 'Two year'],
        'PaperlessBilling': ['Yes', 'No', 'Yes', 'Yes', 'No'],
        'PaymentMethod': ['Electronic check', 'Credit card (automatic)', 'Electronic check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
        'MonthlyCharges': [29.85, 56.95, 20.00, 85.50, 45.25],
        'TotalCharges': ['29.85', '1889.50', ' ', '3860.00', '3242.85'],  # Note: ' ' for new customer
        'Churn': ['No', 'No', 'Yes', 'No', 'No']
    })


class TestTotalChargesFix:
    """Tests for TotalCharges handling."""

    def test_converts_string_to_float(self, sample_raw_data):
        """Test that string TotalCharges is converted to float."""
        df = _fix_total_charges(sample_raw_data.copy())
        assert df['TotalCharges'].dtype == np.float64

    def test_handles_blank_values(self, sample_raw_data):
        """Test that blank TotalCharges values are filled."""
        df = _fix_total_charges(sample_raw_data.copy())
        assert df['TotalCharges'].isna().sum() == 0

    def test_blank_filled_with_monthly(self, sample_raw_data):
        """Test that blank TotalCharges is filled with MonthlyCharges."""
        df = _fix_total_charges(sample_raw_data.copy())
        # Row with blank TotalCharges should have TotalCharges = MonthlyCharges
        blank_row = sample_raw_data[sample_raw_data['TotalCharges'] == ' ']
        if len(blank_row) > 0:
            idx = blank_row.index[0]
            expected = sample_raw_data.loc[idx, 'MonthlyCharges']
            assert df.loc[idx, 'TotalCharges'] == expected


class TestTargetEncoding:
    """Tests for target variable encoding."""

    def test_encodes_yes_no_to_binary(self, sample_raw_data):
        """Test that Yes/No is encoded to 1/0."""
        df = _encode_target(sample_raw_data.copy())
        assert set(df['Churn'].unique()) == {0, 1}

    def test_yes_becomes_1(self, sample_raw_data):
        """Test that Yes is encoded as 1."""
        df = _encode_target(sample_raw_data.copy())
        original_yes_count = (sample_raw_data['Churn'] == 'Yes').sum()
        encoded_1_count = (df['Churn'] == 1).sum()
        assert original_yes_count == encoded_1_count


class TestPreprocessing:
    """Tests for full preprocessing pipeline."""

    def test_drops_customer_id(self, sample_raw_data):
        """Test that customerID is dropped."""
        df = preprocess_data(sample_raw_data.copy())
        assert 'customerID' not in df.columns

    def test_output_has_no_missing_values(self, sample_raw_data):
        """Test that preprocessing removes missing values."""
        df = preprocess_data(sample_raw_data.copy())
        assert df.isnull().sum().sum() == 0

    def test_output_shape(self, sample_raw_data):
        """Test that row count is preserved."""
        df = preprocess_data(sample_raw_data.copy())
        assert len(df) == len(sample_raw_data)


class TestFeatureEngineering:
    """Tests for feature engineering."""

    def test_creates_expected_features(self, sample_raw_data):
        """Test that all expected features are created."""
        df = preprocess_data(sample_raw_data.copy())
        df = engineer_features(df)

        expected = get_engineered_feature_names()
        for feature in expected:
            assert feature in df.columns, f"Missing feature: {feature}"

    def test_tenure_group_values(self, sample_raw_data):
        """Test that tenure_group has valid values."""
        df = preprocess_data(sample_raw_data.copy())
        df = engineer_features(df)

        valid_groups = ['0-12', '13-24', '25-48', '49-72', '72+']
        for group in df['tenure_group'].unique():
            assert group in valid_groups

    def test_binary_features_are_binary(self, sample_raw_data):
        """Test that binary features only have 0 and 1."""
        df = preprocess_data(sample_raw_data.copy())
        df = engineer_features(df)

        binary_features = [
            'is_new_customer', 'high_monthly_charge', 'has_premium_support',
            'has_streaming', 'auto_payment', 'has_family', 'is_senior_alone'
        ]

        for feature in binary_features:
            assert set(df[feature].unique()).issubset({0, 1}), f"{feature} is not binary"

    def test_service_count_range(self, sample_raw_data):
        """Test that service_count is in valid range."""
        df = preprocess_data(sample_raw_data.copy())
        df = engineer_features(df)

        assert df['service_count'].min() >= 0
        assert df['service_count'].max() <= 9  # Max possible services

    def test_contract_value_mapping(self, sample_raw_data):
        """Test contract_value mapping."""
        df = preprocess_data(sample_raw_data.copy())
        df = engineer_features(df)

        assert set(df['contract_value'].unique()).issubset({0, 1, 2})


class TestDataSplitting:
    """Tests for stratified splitting."""

    def test_split_proportions(self, sample_raw_data):
        """Test that splits have approximately correct proportions."""
        df = preprocess_data(sample_raw_data.copy())

        # Need more samples for meaningful split
        df = pd.concat([df] * 20, ignore_index=True)

        train, val, test = create_stratified_splits(df, verbose=False)

        total = len(train) + len(val) + len(test)
        assert abs(len(train) / total - 0.70) < 0.05
        assert abs(len(val) / total - 0.15) < 0.05
        assert abs(len(test) / total - 0.15) < 0.05

    def test_stratification_preserved(self, sample_raw_data):
        """Test that churn ratio is preserved across splits."""
        df = preprocess_data(sample_raw_data.copy())

        # Need more samples
        df = pd.concat([df] * 20, ignore_index=True)

        train, val, test = create_stratified_splits(df, verbose=False)

        overall_rate = df['Churn'].mean()
        train_rate = train['Churn'].mean()
        val_rate = val['Churn'].mean()
        test_rate = test['Churn'].mean()

        # Rates should be within 5% of overall
        assert abs(train_rate - overall_rate) < 0.10
        assert abs(val_rate - overall_rate) < 0.10
        assert abs(test_rate - overall_rate) < 0.10

    def test_no_data_leakage(self, sample_raw_data):
        """Test that there's no overlap between splits."""
        df = preprocess_data(sample_raw_data.copy())
        df = pd.concat([df] * 20, ignore_index=True)
        df['unique_id'] = range(len(df))

        train, val, test = create_stratified_splits(df, verbose=False)

        train_ids = set(train['unique_id'])
        val_ids = set(val['unique_id'])
        test_ids = set(test['unique_id'])

        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
