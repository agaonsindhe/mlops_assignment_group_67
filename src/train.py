"""
This module contains functions to train and evaluate a Linear Regression model
for stock price prediction using historical data.
"""

from datetime import datetime
import mlflow
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from src.preprocess import preprocess_data
from src.utils.grid_search_utils import perform_grid_search
from src.utils.utils import load_config, get_config_path
from src.utils.logging_utils import log_predicted_vs_actual, log_residual_plot
from src.utils.dataset_utils import get_dataset_version, load_data

global input_example, signature, x_test, y_test


def train_and_log_runs(config_path="config.yaml"):
    """
    Train and log three different Ridge regression models with varying hyperparameters and log the best model.
    """
    # Start MLflow Experiment

    experiment_name = f"Stock Price Prediction - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    mlflow.set_experiment(experiment_name)

    # Load configuration
    config = load_config(config_path)

    # Get the dataset path dynamically
    data_path, model_path = get_config_path(config)

    dataset_version = get_dataset_version(data_path)

    # Load and preprocess data
    df = load_data(data_path)

    df = preprocess_data(df)

    # Dynamically select features based on existing columns
    required_features = ['Open', 'High', 'Low', 'Volume', 'Close_ma_3', 'Close_ma_7', 'Close_lag_1']
    features = [col for col in required_features if col in df.columns]
    random_states = [42, 84, 123]
    if not features:
        raise ValueError("No valid features found in the dataset. Check the dataset for missing columns.")

    target = 'Close_pct_change'
    if 'Close_pct_change' not in df.columns:
        df['Close_pct_change'] = df['Close'].pct_change()

    # Drop NaN values and align indices between features and target
    features = df[features].dropna()
    target = df[target].dropna()

    # Align indices to ensure consistent rows
    common_indices = features.index.intersection(target.index)
    features = features.loc[common_indices]
    target = target.loc[common_indices]

    # Define hyperparameters for three runs
    param_grid = {
        "alpha": [0.1, 1.0, 10.0],
        "solver": ["auto", "svd", "cholesky"]
    }

    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=random_states[0])


    # Perform GridSearchCV using your method
    best_model, best_params = perform_grid_search(Ridge(), param_grid, x_train, y_train,x_test, y_test,dataset_version)


if __name__ == "__main__":

    # Run training and evaluation
    train_and_log_runs("config.yaml")
