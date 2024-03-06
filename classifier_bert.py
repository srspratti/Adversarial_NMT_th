from torch import cuda
from torch.autograd import Variable
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertTokenizerFast
from discriminator_cnn_bert import Discriminator_cnn_bert  # Make sure to import your model class correctly
import dill
import logging
import os
import argparse
import options
from datasets import load_dataset
from torch.utils.data import DataLoader


sys.path.append(
    "/u/prattisr/phase-2/all_repos/Adversarial_NMT/neural-machine-translation-using-gan-master"
)
# https://stackoverflow.com/questions/67311527/how-to-set-gpu-count-to-0-using-os-environcuda-visible-devices
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
"""
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
torch.cuda.device_count() # result is 2

os.environ["CUDA_VISIBLE_DEVICES"]="0"
torch.cuda.device_count() # result is 1, using first GPU

os.environ["CUDA_VISIBLE_DEVICES"]="1"
torch.cuda.device_count() # result is 1, using second GPU"""
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

#### Logging ####

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)

parser = argparse.ArgumentParser(description="Adversarial-NMT-BERT")

# Load args
options.add_general_args(parser)
options.add_dataset_args(parser)
options.add_distributed_training_args(parser)
options.add_optimization_args(parser)
options.add_checkpoint_args(parser)
options.add_generator_model_args(parser)
options.add_discriminator_model_args(parser)
options.add_generation_args(parser)

"""
# # Define the device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Initialize tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# def load_model(model_path, use_dill=False):
#     """
#     #Function to load the model.
#     #Set use_dill=True if the model was saved with dill.
#     """
#     if use_dill:
#         return torch.load(open(model_path, 'rb'), pickle_module=dill)
#     else:
#         # Assume args for Discriminator_cnn_bert initialization is provided or modify as needed
#         args = None  # Define this based on your model's requirements
#         model = Discriminator_cnn_bert(args)
#         checkpoint = torch.load(model_path)
#         model.load_state_dict(checkpoint['discriminator_state_dict'])
#         return model


def preprocess(data):
    # Initialize the BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")

    en=list()
    fr=list()
    for element in data['translation']:
        # print("element: ", element)
        en.append(element['en'])
        fr.append(element['fr'] )
    
    # Tokenize the data
    inputs = tokenizer(
        en, truncation=True, padding="max_length", max_length=128
    )
    targets = tokenizer(
        fr, truncation=True, padding="max_length", max_length=128
    )

    # Convert tokens to their corresponding IDs
    input_ids = inputs.input_ids
    target_ids = targets.input_ids

    # Create attention masks
    input_attention_mask = inputs.attention_mask
    target_attention_mask = targets.attention_mask

    return {
        "input_ids": input_ids,
        "attention_mask": input_attention_mask,
        "target_ids": target_ids,
        "target_attention_mask": target_attention_mask,
    }


def predict(model, text):
    """
    #Function to predict using the model.
    """
    model.eval()  # Set model to evaluation mode
    inputs = preprocess(text)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        # Assuming your model's forward method can handle these inputs directly
        # You might need to adjust this based on your model's requirements
        output = model(input_ids, attention_mask)

    return output


def main(args):
    
    use_cuda = torch.cuda.is_available()

    # Set model parameters
    args.encoder_embed_dim = 768 #1000 # changed to 768 to match the BERT model
    args.encoder_layers = 2  # 4
    args.encoder_dropout_out = 0
    args.decoder_embed_dim = 768 #1000 #changed to 768 to match the BERT model
    args.encoder_heads = 2
    args.encoder_ffn_embed_dim = 1000

    args.decoder_heads = 2
    args.decoder_ffn_embed_dim = 1000
    args.decoder_layers = 2  # 4
    args.decoder_out_embed_dim = 1000
    args.decoder_dropout_out = 0
    args.bidirectional = False

    # Loading data using datasets library from the HuggingFace    
    # Get user's home directory
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
    # Loading test dataset
    test_dataset = dataset["test"]
    
    
    # # Path to your saved model
    # model_path = 'path/to/your/saved/model.pt'
    # model = load_model(model_path, use_dill=True)  # Set use_dill based on how you saved your model
    # model.to(device)

    # Path to your saved checkpoint
    saved_model = 'validation_checkpoint_1.pt'
    checkpoints_path = 'checkpoints/bert_dualG/wmt14_en_fr_20k/' + saved_model

    # Load the saved state dictionaries
    checkpoint = torch.load(checkpoints_path)
    
    discriminator_cnn = Discriminator_cnn_bert(args, use_cuda=use_cuda)

    # Load the state dictionaries into the model
    discriminator_cnn.load_state_dict(checkpoint['discriminator_state_dict'])

    discriminator_cnn.eval()  # Set the model to evaluation mode

    # Example text to classify
    text = ["This is a sample sentence.", "Another sample text to evaluate."]
    
    # Prediction
    for t in text:
        output = predict(discriminator_cnn, t)
        print(f"Text: {t}, Prediction: {output}")

if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        logging.warning(f"unknown arguments: {parser.parse_known_args()[1]}")
    main(options)
