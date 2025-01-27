"""
This module contains grid search utility functions for the stock model project.
"""
import pickle
from math import sqrt
import mlflow
from mlflow.models import infer_signature
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.model_selection import GridSearchCV


def perform_grid_search(model, param_grid, x_train, y_train, x_test, y_test, dataset_version, cv=5):
    """
    Perform GridSearchCV, log all runs to MLflow with detailed metrics, model, and dataset version,
    and return the best model and parameters.
    """
    # Initialize GridSearchCV
    grid_search = GridSearchCV(model, param_grid, scoring="neg_mean_squared_error", cv=cv, verbose=1)

    # Fit GridSearchCV
    grid_search.fit(x_train, y_train)

    # Log all runs to MLflow
    for i, params in enumerate(grid_search.cv_results_["params"]):
        with mlflow.start_run(nested=True):

            mlflow.log_param("run_index", i + 1)

            # Retrieve metrics for this run
            mean_mse = -grid_search.cv_results_["mean_test_score"][i]
            std_mse = grid_search.cv_results_["std_test_score"][i]

            # Train model on the entire training set (to compute test metrics)
            model_instance = model.set_params(**params)
            model_instance.fit(x_train, y_train)

            # Evaluate model on test data
            metrics = evaluate_model(model_instance, x_test, y_test)

            # Log hyperparameters
            for key, value in params.items():
                mlflow.log_param(key, value)

            # Log dataset version
            mlflow.log_param("dataset_version", dataset_version)

            # Log metrics
            mlflow.log_metric("mean_mse", mean_mse)
            mlflow.log_metric("std_mse", std_mse)
            mlflow.log_metric("rmse", metrics["rmse"])
            mlflow.log_metric("mae", metrics["mae"])
            mlflow.log_metric("evs", metrics["evs"])
            mlflow.log_metric("r2", metrics["r2"])

            # Log the model
            signature, input_example = infer_mlflow_signature(x_train)
            mlflow.sklearn.log_model(model_instance, "model", input_example=input_example, signature=signature)

            print(f"Logged run with params: {params} and metrics: {metrics}")

    # Retrieve the best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Log the best model separately
    with mlflow.start_run():
        mlflow.log_param("best_alpha", best_params.get("alpha"))
        mlflow.log_param("best_solver", best_params.get("solver"))
        mlflow.log_param("dataset_version", dataset_version)
        mlflow.log_metric("best_mean_mse", -grid_search.best_score_)

        # Evaluate the best model on test data
        best_metrics = evaluate_model(best_model, x_test, y_test)
        mlflow.log_metric("rmse", best_metrics["rmse"])
        mlflow.log_metric("mae", best_metrics["mae"])
        mlflow.log_metric("evs", best_metrics["evs"])
        mlflow.log_metric("r2", best_metrics["r2"])
        with open("model.pkl", "wb") as f:
            pickle.dump(best_model, f)

        # Log the best model
        mlflow.sklearn.log_model(best_model, "best_model", input_example=input_example, signature=signature)

    return best_model, best_params


def infer_mlflow_signature(x_train):
    """
        Infers an MLflow signature for a given training dataset.

        Parameters:
        -----------
        x_train : array-like or DataFrame
            Training dataset where each row is an example and each column is a feature.
            The first row is used to create an input example.

        Returns:
        --------
        signature : mlflow.models.signature.ModelSignature
        input_example : array-like or DataFrame
    """
    input_example = x_train[:1]
    placeholder_model = Ridge(alpha=1.0)
    try:
        placeholder_predictions = placeholder_model.predict(input_example)
    except:
        placeholder_predictions = [0]
    signature = infer_signature(input_example, placeholder_predictions)

    return signature, input_example

def evaluate_model(model, x_test_param, y_test_param):
    """
    Evaluate the model on test data and return metrics.
    """
    y_pred = model.predict(x_test_param)
    mse = mean_squared_error(y_test_param, y_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_test_param, y_pred)
    evs = explained_variance_score(y_test_param, y_pred)
    r2 = r2_score(y_test_param, y_pred)

    return {"rmse": rmse, "mae": mae, "evs": evs, "r2": r2}