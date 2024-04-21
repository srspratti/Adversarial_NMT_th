import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer
import pandas as pd
import sqlite3
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class TranslationDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def read_data_from_db(db_name):
    conn = sqlite3.connect(db_name)
    query = "SELECT src_sentence, translation, ht_or_mt_target FROM balanced_data;"
    data = pd.read_sql_query(query, conn)
    conn.close()
    return data

def predict(model, dataloader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input_ids'].to(model.device)
            labels = batch['labels'].to(model.device)
            outputs = model(inputs)
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.tolist())
            actuals.extend(labels.tolist())
    return predictions, actuals

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

def load_model(model_path, num_labels=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
    model.to(device)
    return model

def main():
    test_data = '/workspace/2024/Adversarial_NMT_th/test_data/wmt14_en_fr_test/standard_classifiers/wmt14_en_fr_test_std_classifiers.csv'  # Adjust this path to your test database
    model_path = '/workspace/2024/git_repo_vastai/roberta_using_balanced_data_train_wmt14_en_fr_1mil_pg_kd_loss_MarianMT_unfreezeonlylmlayer_1mil_20epochs_save_pretrained_with_tokenizer_dict_format_1millimit_v2'  # Adjust to where you saved your trained model

    df_from_db = load_test_data(test_data)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # Prepare the test data
    texts = df_from_db['src'] + " " + df_from_db['ht_mt_target']
    labels = df_from_db['ht_mt_label'].astype(int).tolist()
    encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=512)
    
    test_dataset = TranslationDataset(encodings, labels)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Load the trained model
    model = load_model(model_path, num_labels=2)
    
    # model = load_model(model_path, num_labels=2)

    predictions, actuals = predict(model, test_loader)
    print("Accuracy Score:\n", accuracy_score(actuals, predictions))
    # print("Classification Report:\n", classification_report(actuals, predictions))
    # print("Confusion Matrix:\n", confusion_matrix(actuals, predictions))

if __name__ == "__main__":
    main()
