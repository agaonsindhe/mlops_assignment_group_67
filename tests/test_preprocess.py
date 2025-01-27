from src.preprocess import preprocess_data
from src.utils.utils import load_config, get_config_path, load_data


def test_preprocess_data():
    # Load configuration
    config = load_config("config.yaml")

    # Get the dataset path dynamically
    data_path, model_path = get_config_path(config)

    # Load and preprocess data
    input_data = load_data(data_path)
    required_features = ['Open', 'High', 'Low', 'Volume', 'Close_ma_3', 'Close_ma_7', 'Close_lag_1']

    result = preprocess_data(input_data)
    assert all(feature in input_data.columns for feature in required_features), "Required features are missing in the input data"

