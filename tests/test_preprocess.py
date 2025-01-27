from unittest.mock import patch
from src.preprocess import preprocess_data
from src.utils.utils import load_config, get_config_path, load_data
import pytest
import pandas as pd
import numpy as np

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


@patch("src.preprocess.preprocess_data")
@patch("src.utils.utils.load_data")
@patch("src.utils.utils.get_config_path")
@patch("src.utils.utils.load_config")
def test_preprocess_data(mock_load_config,
    mock_get_config_path,
    mock_load_data,
    mock_preprocess_data,  sample_config,
    sample_data):
    # Mock configuration loading
    mock_load_config.return_value = sample_config

    # Mock config path retrieval
    mock_get_config_path.return_value = ("data/stocks_df.csv", "models/model.pkl")

    # Mock data loading and preprocessing
    mock_load_data.return_value = sample_data
    mock_preprocess_data.return_value = sample_data
    # Load and preprocess data

    required_features = ['Open', 'High', 'Low', 'Volume', 'Close_ma_3', 'Close_ma_7', 'Close_lag_1']

    result = preprocess_data(mock_load_data.return_value )
    assert all(feature in result.columns for feature in required_features), "Required features are missing in the input data"

