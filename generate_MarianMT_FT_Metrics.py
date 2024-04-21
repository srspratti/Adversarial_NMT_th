# import datasets
from datasets import load_dataset
import evaluate
import os
import torch
import dill
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

def main():

    # Get user's home directory
    import os
    home = os.path.expanduser("~")

    # Define the path of the cache directory
    cache_dir = os.path.join(home, ".cache", "huggingface", "datasets")

    # Define the name and configuration of the dataset
    dataset_name = "wmt14"
    config_name = "fr-en"

    # Build the path for the specific dataset configuration
    dataset_config_path = os.path.join(cache_dir, dataset_name, config_name)

    print(f"Checking cache at: {dataset_config_path}")

    # Check if the dataset configuration is already cached
    if os.path.exists(dataset_config_path) and len(os.listdir(dataset_config_path)) > 0:
        print("Dataset already downloaded, loading from cache.")
        # If the dataset is already downloaded, load it from the cache directory
        dataset = load_dataset(dataset_name, config_name, cache_dir=cache_dir)
    else:
        print("Downloading the dataset.")
        # Download the dataset and specify the cache directory
        dataset = load_dataset(dataset_name, config_name, cache_dir=cache_dir)

    # Here, you should adjust the loading of subsets to avoid redundant downloads or loading.
    # Load 50k rows of the train dataset
    # train_dataset = dataset["train"].select(range(100020))
    # train_dataset = dataset["train"].select(range(600))

    # Keep the full valid and test datasets
    valid_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    texts =[]
    labels = []
    for element in test_dataset["translation"]:
            # print("element: ", element)
            texts.append(element["en"])
            labels.append(element["fr"])
    
    metric = evaluate.load("sacrebleu")
    getpwd = os.getcwd()


    file_path_en = os.path.join(getpwd, "original_english_mm_v2.txt")
    # file_path = "/path/to/translations.txt"

    # Open the file in write mode
    with open(file_path_en, "w") as file:
        # Write each translation to the file
        for text in texts:
            file.write(text + "\n")

    
    file_path_fr = os.path.join(getpwd, "original_french_mm_v2.txt")
    # file_path = "/path/to/translations.txt"

    # Open the file in write mode
    with open(file_path_fr, "w") as file:
        # Write each translation to the file
        for label in labels:
            file.write(label + "\n")
    
    checkpoint_path_generator = '/home/paperspace/google_drive_v1/Research_Thesis/2024/git_repo/checkpoints/bert_dualG/wmt14_en_fr_1mil_pg_kd_loss_MarianMT_unfreezeonlylmlayer_600sents_dedbug_spcChars__save_pretrained_v2/best_generator_at_1.pt'
    checkpoint_path_tokenizer = "/home/paperspace/google_drive_v1/Research_Thesis/2024/git_repo/checkpoints/bert_dualG/wmt14_en_fr_1mil_pg_kd_loss_MarianMT_unfreezeonlylmlayer_600sents_dedbug_spcChars__save_pretrained_v2/best_generator_tokenizer_save_pretrained_at_1"
    translations_generated_filename = "translated_french_by_MarianMT_FT_600sents.txt"
    
    # Load the entire model directly
    generator2_checkpoint = torch.load(open(checkpoint_path_generator, "rb"), pickle_module=dill)

    # generator2_train # Extract the underlying model from the DataParallel wrapper
    generator2_checkpoint = generator2_checkpoint.module if isinstance(generator2_checkpoint, torch.nn.DataParallel) else generator2_checkpoint

    # Check if CUDA is available and then set the default device to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    generator2_checkpoint.eval()
    generator2_checkpoint.to('device')
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path_tokenizer)


    translations= []

    # Assuming 'texts' is defined elsewhere and contains the English sentences to be translated
    for idx, text in tqdm(enumerate(texts), desc="Translating", total=len(texts)):
        inputs = tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt").input_ids.to(device)
        
        # Generate the outputs using the model
        outputs = generator2_checkpoint.generate(inputs, max_length=60, num_beams=5, early_stopping=True)
        
        # Decode the generated IDs to text
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        translations.append(translation)

    # Save the translations to a text file
    getpwd = os.getcwd()
    file_path = os.path.join(getpwd, translations_generated_filename)

    with open(file_path, "w") as file:
        for translation in translations:
            file.write(translation + "\n")



if __name__ == "__main__":
    main()