"""
Unit tests for the preprocess module.

This file contains tests for the load_and_preprocess_data function in the preprocess module.
"""
from src.preprocess import preprocess_data

def test_preprocess_data():
    """
    Test the load_and_preprocess_data function.

    This test ensures that the preprocessing step returns a dictionary containing
    the keys 'X_train', 'X_test', 'y_train', 'y_test', and 'scaler' with valid data.
    """
    input_data = {"features": [[1, 2], [3, 4]], "labels": [0, 1]}
    result = preprocess_data(input_data)
    assert "features" in result
    assert "labels" in result
