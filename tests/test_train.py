import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from src.train import train_and_log_runs

@pytest.fixture
def sample_config():
    return {
        "dataset": {
            "preferred_path": "data/stocks_df.csv",
            "fallback_path": "data/stock_data_sample.csv"
        },
        "model": {
            "path": "model.pkl"
        }
    }

@pytest.fixture
def sample_data():
    data = {
        "Date": ['2008-03-04', '2008-03-19', '2008-03-17', '2008-03-18', '2008-03-05'],
        "Stock": ['20MICRONS', 'RELCAPITAL', 'NITIRAJ', 'LTI', 'KAMATHOTEL'],
        "Open": [100, 102, 104, 106, 108],
        "High": [101, 103, 105, 107, 109],
        "Low": [99, 101, 103, 105, 107],
        "Volume": [1000, 1100, 1200, 1300, 1400],
        "Close": [100, 102, 104, 106, 108],
        "Close_ma_3": [np.nan, np.nan, 102, 104, 106],
        "Close_ma_7": [np.nan, np.nan, np.nan, np.nan, np.nan],
        "Close_lag_1": [np.nan, 100, 102, 104, 106],
        "Change Pct": [-3, -2, 2, 1.1, -2.1],
    }
    df = pd.DataFrame(data)
    return df


@patch("src.train.mlflow.sklearn.log_model")
@patch("src.train.mlflow.log_metric")
@patch("src.train.mlflow.log_param")
@patch("src.train.mlflow.start_run")
@patch("src.preprocess.preprocess_data")
@patch("src.utils.utils.load_data")
@patch("src.utils.utils.get_config_path")
@patch("src.utils.utils.load_config")
def test_train_and_log_runs(
    mock_load_config,
    mock_get_config_path,
    mock_load_data,
    mock_preprocess_data,
    mock_start_run,
    mock_log_param,
    mock_log_metric,
    mock_log_model,
    sample_config,
    sample_data,
):
    # Mock configuration loading
    mock_load_config.return_value = sample_config

    # Mock config path retrieval
    mock_get_config_path.return_value = ("data/stocks_df.csv", "models/model.pkl")

    # Mock data loading and preprocessing
    mock_load_data.return_value = sample_data
    mock_preprocess_data.return_value = sample_data

    # Call the function
    train_and_log_runs()

    # Assertions
    assert mock_start_run.call_count == 4  # 3 runs + 1 best model
    assert mock_log_param.call_count > 0
    assert mock_log_metric.call_count > 0

