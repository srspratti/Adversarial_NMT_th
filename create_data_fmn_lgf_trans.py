import pandas as pd
import sqlite3

import pandas as pd
import sqlite3
import os

def write_df_to_db(db_name, df):
    # Connect to the database
    conn = sqlite3.connect(db_name)

    # Create a cursor object
    cur = conn.cursor()

    # Drop the table if it already exists
    cur.execute("DROP TABLE IF EXISTS balanced_data")

    # Create the table
    cur.execute("""CREATE TABLE balanced_data (
        epoch INTEGER,
        src_sentence TEXT,
        translation TEXT,
        ht_or_mt_target INTEGER
    )""")

    # Insert the data into the table
    for row in df.itertuples():
        cur.execute("""INSERT INTO balanced_data (epoch, src_sentence, translation, ht_or_mt_target)
        VALUES (?, ?, ?, ?)""",
        (row.epoch, row.src_sentence, row.translation, row.ht_or_mt_target))

    # Commit the changes to the database
    conn.commit()

    # Close the connection
    conn.close()

def fetch_balanced_data_train(db_name, epochs):
    """
    Fetches a balanced dataset from the database, with an equal number of human and machine translations.
    
    Args:
    - db_name: The database name.
    - epochs: Total number of epochs.
    - N: Number of samples to fetch per category per epoch.
    """
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    
    # Getting the row count : 
    query = """
    select count(*) from translations_train;
    """
    c.execute(query)
    row_count = c.fetchall()
    # row_count = 9
    print("row count: ", row_count)
    print("type of row_count: ", type(row_count[0][0]))
    row_count = row_count[0][0]

    # Placeholder lists for data
    data = []
    
    for epoch_i in range(1, epochs+1):
        # Calculate the limit for each category based on N
        limit_human = int(row_count / 2)
        limit_machine = int(row_count / 4)  # Assuming you want half for G2 and half for G1
        print("limit machine ", limit_machine)
        print("limit human ", limit_human)

        # Fetch an equal number of human and machine translations for the epoch
        query_human = """
        SELECT epoch_i_list, src_sentences_converted_logging_org, tgt_sentences_converted_logging_org, '1' AS ht_or_mt_target
        FROM translations_train
        WHERE epoch_i_list=?
        ORDER BY RANDOM()
        LIMIT ?
        """
        
        query_machine_G2 = """
        SELECT epoch_i_list, src_sentences_converted_logging_org, fake_tgt_sentences_converted_logging_G2_train, '0'
        FROM translations_train
        WHERE epoch_i_list=?
        ORDER BY RANDOM()
        LIMIT ?
        """
        
        query_machine_G1 = """
        SELECT epoch_i_list, src_sentences_converted_logging_org, fake_tgt_sentences_G1_pretrain_converted_logging, '0'
        FROM translations_train
        WHERE epoch_i_list=?
        ORDER BY RANDOM()
        LIMIT ?
        """
        
        # Execute queries and append results
        c.execute(query_human, (epoch_i, limit_human))
        human_translations = c.fetchall()
        print("length of human translations fetched from DB: ", len(human_translations))
        print("type of human_translations ", type(human_translations))
        print("human_translations ", human_translations)
        data.extend(human_translations)
        
        c.execute(query_machine_G2, (epoch_i,limit_machine))
        machine_translations_G2 = c.fetchall()
        print("length of machine translations by G2-Train() fetched from DB: ", len(machine_translations_G2))
        print("type of machine_translations_G2 ", type(machine_translations_G2))
        print("machine_translations_G2 ", machine_translations_G2)
        data.extend(machine_translations_G2)
        
        c.execute(query_machine_G1, (epoch_i, limit_machine))
        machine_translations_G1 = c.fetchall()
        print("length of machine translations by G1-PreTrain() fetched from DB: ", len(machine_translations_G1))
        print("type of machine_translations_G1 ", type(machine_translations_G1))
        print("machine_translations_G1 ", machine_translations_G1)
        data.extend(machine_translations_G1)
    
    conn.close()
    
    # Create a DataFrame
    df = pd.DataFrame(data, columns=['epoch', 'src_sentence', 'translation', 'ht_or_mt_target'])
    return df

# Usage
# db_name = 'your_database.db'
# epochs = 2  # Example value
# df = fetch_balanced_data_train(db_name, epochs)
def main():
    
    # db_name = os.getcwd() + '/translations.db'
    db_name = '/home/paperspace/google_drive_v1/Research_Thesis/2024/git_repo/translations_wmt14_en_fr_1mil_pg_kd_loss_MarianMT_unfreezeonlylmlayer_1mil_20epochs_save_pretrained_with_tokenizer_dict_format.db'
    print("db_name: ", db_name)
    epochs = 2  # Example value
    df = fetch_balanced_data_train(db_name, epochs)
    print("df: ", df)
    df.to_csv('data_train_wmt14_en_fr_1mil_pg_kd_loss_MarianMT_unfreezeonlylmlayer_1mil_20epochs_save_pretrained_with_tokenizer_dict_format.csv', index=False)
    print("df shape: ", df.shape)

    # writing the df to db
    db_name_train_data = os.getcwd() + '/balanced_data_train_wmt14_en_fr_1mil_pg_kd_loss_MarianMT_unfreezeonlylmlayer_1mil_20epochs_save_pretrained_with_tokenizer_dict_format.db'
    write_df_to_db(db_name_train_data, df)


if __name__ == "__main__":
    main()