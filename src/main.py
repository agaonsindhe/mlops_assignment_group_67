from src.preprocess import preprocess_data
from src.train import train_model

def main():
    # Example: Load data
    data = {"features": [[1, 2], [3, 4]], "labels": [0, 1]}
    
    # Preprocess the data
    processed_data = preprocess_data(data)
    
    # Train a simple model
    train_model(processed_data)

if __name__ == "__main__":
    main()
