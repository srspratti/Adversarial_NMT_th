import pandas as pd
import sqlite3
import os


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from transformers import RobertaTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset, DataLoader
import torch

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

def main():

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("GPU not found, using CPU instead.")

    print("Hello, world! This is the train_svm.py script.")
    # train_db_path = os.getcwd() + "/balanced_data_train.db"
    train_db_path = '/workspace/2024/git_repo_vastai/balanced_data_train_wmt14_en_fr_1mil_pg_kd_loss_MarianMT_unfreezeonlylmlayer_1mil_20epochs_save_pretrained_with_tokenizer_dict_format_1millimit_v2.db'
    # train_db_path = '/workspace/2024/git_repo_vastai/balanced_data_train_wmt14_en_fr_1mil_pg_kd_loss_MarianMT_unfreezeonlylmlayer_1mil_20epochs_save_pretrained_with_tokenizer_dict_format_sm_20k.db'
    df_from_db = read_data_from_db(train_db_path)
    print("df_from_db head: ", df_from_db.head())

    # Tokenize the input texts
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Preparing the dataset
    texts = df_from_db['src_sentence'] + " " + df_from_db['translation']  # Combining source and translation for input
    labels = df_from_db['ht_or_mt_target'].tolist()

    # Tokenize texts
    encodings = tokenizer(texts.to_list(), truncation=True, padding=True, max_length=512)

    # Create dataset
    dataset = TranslationDataset(encodings, labels)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Define training arguments
    training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=8,  # Adjust based on your GPU's memory
    per_device_eval_batch_size=8,   # Adjust based on your GPU's memory
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    load_best_model_at_end=True,
    evaluation_strategy="steps",  # Evaluate every `logging_steps`
    logging_steps=50,  # Log metrics every 50 steps
    fp16=True,  # Use mixed precision
    )

    # training_args.fp16 = True  # Requires Nvidia Apex or PyTorch >= 1.6

    # Initialize the model
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

    # Initialize the trainer
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    trainer.evaluate()

    trainer.save_model("/workspace/2024/git_repo_vastai/roberta_using_balanced_data_train_wmt14_en_fr_1mil_pg_kd_loss_MarianMT_unfreezeonlylmlayer_1mil_20epochs_save_pretrained_with_tokenizer_dict_format_1millimit_v2")
    # trainer.save_model("/workspace/2024/git_repo_vastai/roberta_balanced_data_train_wmt14_en_fr_1mil_pg_kd_loss_MarianMT_unfreezeonlylmlayer_1mil_20epochs_save_pretrained_with_tokenizer_dict_format_sm_20k.db")

if __name__ == "__main__":
    main()