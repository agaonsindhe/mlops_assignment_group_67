# Stock Price Prediction

This project trains a model to predict the closing price of stocks based on historical data from the India Stock Market dataset.

## Features
- Data preprocessing with lagged features.
- Training a linear regression model to predict closing prices.
- CI/CD pipeline using GitHub Actions.

## Dataset
The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/andrewmvd/india-stock-market).

## Directory Structure
mlops_assignment_group_67/
├── src/
│   ├── main.py
│   ├── preprocess.py
│   └── train.py
├── tests/
│   ├── test_preprocess.py
│   └── test_train.py
├── data/
│   ├── india_stock_market.csv
│   └── placeholder.txt
├── .github/
│   └── workflows/
│       └── ci_pipeline.yml
├── Dockerfile
├── requirements.txt
├── README.md

- `src/`: Contains the main application code.
- `tests/`: Unit tests for the application.
- `.github/workflows/`: CI/CD pipeline definition.
- `data/`: Directory for storing the dataset.
- `requirements.txt`: Dependencies for the project.

## Getting Started
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Add the dataset to the `data/` directory.
4. Run the application: `python src/main.py`.
5. Test the pipeline by committing changes to GitHub.

## CI/CD Pipeline
The pipeline includes:
- **Linting**: Ensures code style consistency.
- **Testing**: Runs unit tests to validate functionality.
