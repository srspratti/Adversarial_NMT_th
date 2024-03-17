import pandas as pd
from joblib import load

def load_test_data(csv_file_path):
    """
    Load test data from a CSV file.
    
    Args:
    - csv_file_path: The path to the .csv file containing the test data.
    
    Returns:
    A pandas DataFrame containing the test data.
    """
    test_df = pd.read_csv(csv_file_path)
    return test_df

def preprocess_test_data(test_df):
    """
    Preprocess the test data as per the model's training data requirements.
    
    Args:
    - test_df: The test data as a pandas DataFrame.
    
    Returns:
    The preprocessed test data.
    """
    # Assuming the model was trained on combined source and translation text
    test_df['combined_text'] = test_df['src_sentence'] + " " + test_df['translation']
    return test_df['combined_text']

def predict_with_saved_model(model_path, test_data):
    """
    Predict the classes of the test data using the saved model.
    
    Args:
    - model_path: The path to the saved model.
    - test_data: The preprocessed test data.
    
    Returns:
    The predicted classes.
    """
    model = load(model_path)
    predictions = model.predict(test_data)
    return predictions

def main():
    model_path = "path/to/your/saved/svm_model_sm_grid.joblib"  # Update with the actual path to your saved model
    csv_file_path = "path/to/your/test_data.csv"  # Update with the actual path to your test data CSV file
    
    # Load and preprocess the test data
    test_df = load_test_data(csv_file_path)
    preprocessed_test_data = preprocess_test_data(test_df)
    
    # Predict with the saved model
    predictions = predict_with_saved_model(model_path, preprocessed_test_data)
    
    # Output the predictions
    print("Predictions:", predictions)

if __name__ == "__main__":
    main()