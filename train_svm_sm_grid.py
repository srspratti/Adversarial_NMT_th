import pandas as pd
import sqlite3
import os


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from joblib import dump,load

def read_data_from_db(db_name):
    """
    Reads the data from the database and returns it as a pandas dataframe.
    
    Args:
    - db_name: The database name.
    """
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    
    # Getting the row count : 
    query = """
    select count(*) from balanced_data;
    """
    c.execute(query)
    row_count = c.fetchall()
    print("row count: ", row_count)
    print("type of row_count: ", type(row_count[0][0]))
    row_count = row_count[0][0]
    
    # Fetching the data from the database
    query = """
    select src_sentence, translation, ht_or_mt_target from balanced_data;
    """
    c.execute(query)
    data = c.fetchall()
    print("data: ", data)
    print("type of data: ", type(data))
    print("len of data: ", len(data))
    print("type of data[0]: ", type(data[0]))
    print("data[0]: ", data[0])
    
    # Creating a dataframe
    df = pd.DataFrame(data, columns=["src_sentence", "translation", "ht_or_mt_target"])
    
    # Closing the connection
    conn.close()
    
    return df


def main():

    print("Hello, world! This is the train_svm.py script.")
    train_db_path = os.getcwd() + "/balanced_data_train.db"
    df_from_db = read_data_from_db(train_db_path)
    print("df_from_db head: ", df_from_db.head())

    # Combine source and translated sentences into a single feature for simplicity
    df_from_db['combined_text'] = df_from_db['src_sentence'] + " " + df_from_db['translation']

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(df_from_db['combined_text'], df_from_db['ht_or_mt_target'], test_size=0.2, random_state=42)

    # Create a pipeline that first vectorizes the text and then applies SVM
    pipeline = make_pipeline(TfidfVectorizer(), SVC(kernel='linear'))

    # Setup the GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid={'svc__C': [0.1, 1, 10, 100]}, cv=5, n_jobs=-1, verbose=1)

    # Train the SVM model
    grid_search.fit(X_train, y_train)

    # dumping the model
    model_path = os.getcwd() + "/svm_model_sm_grid.joblib"
    dump(grid_search.best_estimator_, model_path)
    print(f"Model saved at {model_path}")

    # Evaluate the model
    predictions = grid_search.predict(X_test)
    print(f"SVM Accuracy: {accuracy_score(y_test, predictions)}")

if __name__ == "__main__":
    main()