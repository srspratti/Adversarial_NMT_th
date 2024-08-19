import torch
import datasets
import os
from datasets import load_dataset  # Import load_dataset from Hugging Face

def main():

    # Define the path of the cache directory
    home = os.path.expanduser("~")
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

    # Keep the full valid and test datasets
    valid_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    # Extract English and French translations
    texts = []
    labels = []
    for element in test_dataset["translation"]:
        texts.append(element["en"])
        labels.append(element["fr"])


    ### Generator Teacher - Translations and evaluating bleu score for the test translations

    # Load the pretrained model using torch.hub
    fairseq_PT_GeneratorTeacher = torch.hub.load('pytorch/fairseq', 'transformer.wmt14.en-fr', tokenizer='moses', bpe='subword_nmt')

    # Set the model to evaluation mode
    fairseq_PT_GeneratorTeacher.eval()

    # Example batch of input sentences (English)
    # input_sentences = [
    #     "The weather is nice today.",
    #     "I am learning how to code.",
    #     "This is a test sentence.",
    # ]

    input_sentences = []
    input_sentences = texts

    # Tokenize and encode the input sentences as a batch
    tokens = [fairseq_PT_GeneratorTeacher.encode(sentence) for sentence in input_sentences]

    # Pad the tokenized inputs to the same length for batch processing
    # max_length = max(len(t) for t in tokens)
    # max_length = 128

    # tokens_padded = [torch.cat([t, torch.tensor([fairseq_PT_GeneratorTeacher.task.source_dictionary.pad()]) * (max_length - len(t))]) for t in tokens]

    # # Convert the list of tensors to a single tensor (batch)
    # tokens_batch = torch.stack(tokens_padded)

    # # Generate translations
    # with torch.no_grad():  # Disable gradient calculation for inference
    #     translations = fairseq_PT_GeneratorTeacher.generate(tokens_batch)

    # # Decode the translations
    # translated_sentences = [fairseq_PT_GeneratorTeacher.decode(t) for t in translations]

    # Translate the sentences with a maximum length of 128 tokens
    translated_sentences_from_fairseq_PT_GeneratorTeacher = fairseq_PT_GeneratorTeacher.translate(input_sentences, max_len_a=0, max_len_b=128)

    # Print the translated sentences
    # for sentence in translated_sentences_from_fairseq_PT_GeneratorTeacher:
    #     print(sentence)

    # Now calculate the metrics
    # Load the evaluation metrics
    import evaluate
    bleu_metric = evaluate.load("sacrebleu")
    meteor_metric = evaluate.load("meteor")
    rouge_metric = evaluate.load("rouge")
    ter_metric = evaluate.load("ter")
    comet_metric = evaluate.load("comet")

    result_batch_fairseq_PT_GeneratorTeacher = []

    result_batch_fairseq_PT_GeneratorTeacher = {
    "bleu": bleu_metric.compute(predictions=translated_sentences_from_fairseq_PT_GeneratorTeacher, references=labels)["score"],
    "meteor": meteor_metric.compute(predictions=translated_sentences_from_fairseq_PT_GeneratorTeacher, references=labels)["meteor"],
    "rouge": rouge_metric.compute(predictions=translated_sentences_from_fairseq_PT_GeneratorTeacher, references=labels),
    "ter": ter_metric.compute(predictions=translated_sentences_from_fairseq_PT_GeneratorTeacher, references=labels)["score"],
    "comet": comet_metric.compute(predictions=translated_sentences_from_fairseq_PT_GeneratorTeacher, references=labels, sources=texts)["mean_score"]
    }

    result_batch_fairseq_PT_GeneratorTeacher_path = os.path.join(os.getcwd(), "checkpoints","bert_dualG","fairseq_pretrained_eval_results","result_batch_fairseq_PT_GeneratorTeacher.txt")
    with open( result_batch_fairseq_PT_GeneratorTeacher_path,"w") as f:
        f.write("BLEU Score: " + str(result_batch_fairseq_PT_GeneratorTeacher["bleu"]) + "\n")
        f.write("METEOR Score: " + str(result_batch_fairseq_PT_GeneratorTeacher["meteor"]) + "\n")
        f.write("ROUGE Scores: " + str(result_batch_fairseq_PT_GeneratorTeacher["rouge"]) + "\n")
        f.write("TER Score: " + str(result_batch_fairseq_PT_GeneratorTeacher["ter"]) + "\n")
        f.write("COMET Score: " + str(result_batch_fairseq_PT_GeneratorTeacher["comet"]) + "\n")

    # Save the translations to a text file - translations
   
    file_path = os.path.join(os.getcwd(), "checkpoints","bert_dualG","fairseq_pretrained_eval_results", "translated_sentences_from_fairseq_PT_GeneratorTeacher.txt")
    with open(file_path, "w") as file:
        for translation in translated_sentences_from_fairseq_PT_GeneratorTeacher:
            file.write(translation + "\n")


    getpwd = os.getcwd()
    file_path_en = os.path.join(getpwd, "checkpoints","bert_dualG","fairseq_pretrained_eval_results","original_english_translations_fairseq_PT_GeneratorTeacher.txt")
    # file_path = "/path/to/translations.txt"

    # Open the file in write mode
    with open(file_path_en, "w") as file:
        # Write each translation to the file
        for text in texts:
            file.write(text + "\n")


    file_path_fr = os.path.join(getpwd, "checkpoints","bert_dualG","fairseq_pretrained_eval_results","original_french_translations_fairseq_PT_GeneratorTeacher.txt")
    # file_path = "/path/to/translations.txt"

    # Open the file in write mode
    with open(file_path_fr, "w") as file:
        # Write each translation to the file
        for label in labels:
            file.write(label + "\n")

if __name__ == "__main__":
    main()

