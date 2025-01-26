from src.preprocess import preprocess_data

def test_preprocess_data():
    input_data = {"features": [[1, 2], [3, 4]], "labels": [0, 1]}
    result = preprocess_data(input_data)
    assert "features" in result
    assert "labels" in result
