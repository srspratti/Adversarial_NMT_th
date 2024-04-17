import pandas as pd
from joblib import load
import os
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

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
    # test_df['combined_text'] = test_df['src_sentence'] + " " + test_df['translation']
    # test_df['combined_text'] = test_df['src'] + " " + test_df['target']
    test_df['combined_text'] = test_df['src'] + " " + test_df['ht_mt_target']
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

    getpwd = os.getcwd()
    # testdata_folder = "/test_data/wmt14_en_fr_test/standard_classifiers"
    # model_path = getpwd + "/svm_model_sm_grid.joblib"  # Update with the actual path to your saved model
    model_path = '/workspace/2024/git_repo_vastai/svm_model_sm_100k.joblib'
    # test_data = getpwd + testdata_folder + "wmt14_en_fr_test_std_classifiers.csv"  # Update with the actual path to your test data CSV file
    test_data = '/workspace/2024/Adversarial_NMT_th/test_data/wmt14_en_fr_test/standard_classifiers/wmt14_en_fr_test_std_classifiers.csv'

    # Load and preprocess the test data
    test_df = load_test_data(test_data)
    preprocessed_test_data = preprocess_test_data(test_df)
    
    # Predict with the saved model
    predictions = predict_with_saved_model(model_path, preprocessed_test_data)
    
    # Output the predictions
    print("Predictions:", predictions)

     # Extract the true labels
    true_labels = test_df['ht_mt_label'].values  # Ensure 'true_label' matches the column name in your CSV
    
    # Calculate and print the metrics
    print("Accuracy Score:", accuracy_score(true_labels, predictions))
    print("\nClassification Report:\n", classification_report(true_labels, predictions))
    print("\nConfusion Matrix:\n", confusion_matrix(true_labels, predictions))

if __name__ == "__main__":
    main()