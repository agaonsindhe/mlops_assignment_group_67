"""
Model training module.

This module trains a regression model and evaluates its performance.
"""
def train_model(data):
    """
    Train a linear regression model and evaluate it.

    Args:
        data (dict): Preprocessed data including training and testing sets.

    Returns:
        tuple: Trained model and evaluation metrics.
    """
    print("Training model...")
    # Example training logic
    print(f"Features: {data['features']}")
    print(f"Labels: {data['labels']}")
