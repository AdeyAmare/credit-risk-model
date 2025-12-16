import pandas as pd
import pytest
from src.target_engineering import ProxyTargetGenerator  # replace with actual import
from src.data_processing import TimeFeaturesExtractor, FeatureEngineeringPipeline


# -----------------------------
# Sample data fixture
# -----------------------------
@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'CustomerId': [1, 1, 2, 2, 3, 3],
        'TransactionStartTime': [
            '2025-12-01', '2025-12-05',
            '2025-11-20', '2025-12-10',
            '2025-12-15', '2025-12-16'
        ],
        'Value': [100, 200, 150, 50, 300, 400],
        'Category': ['A', 'B', 'A', 'B', 'C', 'A']
    })


# -----------------------------
# Test 1: RFM calculation
# -----------------------------
def test_calculate_rfm(sample_data):
    generator = ProxyTargetGenerator()
    rfm = generator.calculate_rfm(sample_data)
    assert isinstance(rfm, pd.DataFrame)
    assert all(col in rfm.columns for col in ['CustomerId', 'recency', 'frequency', 'monetary'])
    assert rfm.shape[0] == 3  # 3 unique customers


# -----------------------------
# Test 2: Full pipeline
# -----------------------------
def test_generate(sample_data):
    generator = ProxyTargetGenerator(n_clusters=2)
    high_risk_df = generator.generate(sample_data)
    assert isinstance(high_risk_df, pd.DataFrame)
    assert all(col in high_risk_df.columns for col in ['CustomerId', 'is_high_risk'])
    assert high_risk_df.shape[0] == 3
    assert set(high_risk_df['is_high_risk'].unique()).issubset({0, 1})


# -----------------------------
# Test 3: TimeFeaturesExtractor
# -----------------------------
def test_time_features_extractor(sample_data):
    extractor = TimeFeaturesExtractor(datetime_col='TransactionStartTime')
    transformed = extractor.fit_transform(sample_data)
    assert isinstance(transformed, pd.DataFrame)
    assert all(col in transformed.columns for col in [
        'txn_hour', 'txn_day', 'txn_month', 'txn_year'
    ])


# -----------------------------
# Test 4: Full FeatureEngineeringPipeline
# -----------------------------
def test_feature_engineering_pipeline(sample_data):
    pipeline = FeatureEngineeringPipeline(
        numeric_cols=['Value'],
        categorical_cols=['Category'],
        top_k=2
    )
    pipeline.fit(sample_data)
    transformed = pipeline.transform(sample_data)
    assert isinstance(transformed, pd.DataFrame)
    assert 'CustomerId' in transformed.columns
    # Check that numeric columns are present
    assert any('value' in col.lower() for col in transformed.columns)