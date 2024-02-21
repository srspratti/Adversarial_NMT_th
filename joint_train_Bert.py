# Importing required libraries
import torch
from torch import cuda
from torch.autograd import Variable
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertTokenizerFast
import datasets
from datasets import load_dataset
from torch.utils.data import DataLoader

# importing other required libraries
import argparse
import logging
import math
import dill
import os
import options
import random
import sys
import numpy as np
from collections import OrderedDict

import data
import utils
from meters import AverageMeter
from PGLoss import PGLoss
from tqdm import tqdm
from dictionary import Dictionary
import re
import subprocess

# Importing Generator and Discriminator class methods
from generator_tf_bert import TransformerModel_bert
from discriminator_cnn_bert import Discriminator_cnn_bert

# CUDA multiple-GPU configuration


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
    
    """
    #get user's home directory
    home = os.path.expanduser("~")
    #Define the path of the cache directory
    cache_dir = ".cache/huggingface/datasets"
    
    #Define the name of the dataset
    dataset_name = "wmt14"
    
    # check if the dataset is already downloaded
    dataset_path = os.path.join(home, cache_dir, dataset_name)
    print("dataset_path: ", dataset_path)
    if os.path.exists(dataset_path):
        print("Dataset already downloaded") # if the dataset is already downloaded
        dataset = load_dataset("wmt14", "fr-en",  cache_dir=dataset_path)
    else:
        print("Downloading the dataset")
        # Download the dataset
        dataset = load_dataset("wmt14", "fr-en")
        # dataset = datasets.load_dataset('wmt14', 'fr-en', cache_dir='path_to_cache_dir', download_mode='force_redownload') # if any error occurs, use this line of code
        # dataset = load_dataset("wmt14", "en-fr", split='train[:1%]', streaming=True) # Only loading 1% of train dataset for proto-typing purposes

    # Load 50k rows of the train dataset
    train_dataset = load_dataset("wmt14", "fr-en", split="train[:50000]")

    # Keep the full valid and test datasets
    valid_dataset = dataset["validation"]
    test_dataset = dataset["test"]
    """
    
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
    # Load 50k rows of the train dataset
    train_dataset = dataset['train'].select(range(5000))

    # Keep the full valid and test datasets
    valid_dataset = dataset["validation"]
    test_dataset = dataset["test"]
    
    
    # Loading Bert Model
    bert_model = "bert-base-multilingual-cased"

    # Pre-processing the data

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

    # print(train_dataset[0])
    # tokenized_datasets = dataset.map(tokenize_function, batched=True) # using the other berttokenizer map function
    tokenized_train_datasets = train_dataset.map(
        preprocess, batched=True
    )  # Using the bertFaSTtOKENIZER MAp function
    tokenized_valid_datasets = valid_dataset.map(
        preprocess, batched=True
    )  # Using the bertFaSTtOKENIZER MAp function
    tokenized_test_datasets = test_dataset.map(
        preprocess, batched=True
    )  # Using the bertFaSTtOKENIZER MAp function

    #### --------------------Loading G2 - Generator in Train() in GAN--------------------#####

    generator2_train = TransformerModel_bert(args, use_cuda=use_cuda)

    #### --------------------Loading D - Discriminator in Train() in GAN --------------------#####

    discriminator_cnn = Discriminator_cnn_bert(args, use_cuda=use_cuda)

    ##### ---- Other set-up before starting JOINT TRANING-------############

    if use_cuda:
        if torch.cuda.device_count() > 1:
            discriminator_cnn = torch.nn.DataParallel(discriminator_cnn).cuda()
            generator2_train = torch.nn.DataParallel(generator2_train).cuda()
        else:
            generator2_train.cuda()
            discriminator_cnn.cuda()
    else:
        discriminator_cnn.cpu()
        generator2_train.cpu()

    # adversarial training checkpoints saving path
    if not os.path.exists(
        "checkpoints/bert/wmt14_en_fr_50k"
    ):
        os.makedirs(
            "checkpoints/bert/wmt14_en_fr_50k"
        )
    checkpoints_path = (
        "checkpoints/bert/wmt14_en_fr_50k/"
    )

    ## Define loss functions for the generator and the Discriminator
    g_criterion = torch.nn.NLLLoss(reduction="sum", ignore_index=0)
    d_criterion = torch.nn.BCELoss()
    pg_criterion = PGLoss(size_average=True, reduce=True)

    #### ----------------------------------JOINT TRAINING --------------------#####

    # Define the optimizers
    optimizer_g = torch.optim.Adam(generator2_train.parameters(), lr=0.001)
    optimizer_d = torch.optim.Adam(discriminator_cnn.parameters(), lr=0.001)

    # Start the training loop
    for epoch_i in tqdm(range(1, args.epochs + 1)):
        logging.info("At {0}-th epoch.".format(epoch_i))

        # ------------------Creating dataloader for train
        # Define the batch size
        batch_size = args.joint_batch_size

        # Converting the processed data to PyTorch Tensors
        tokenized_train_datasets.set_format(
            type="torch",
            columns=[
                "input_ids",
                "attention_mask",
                "target_ids",
                "target_attention_mask",
            ],
        )

        # Create a DataLoader for the train data
        train_dataloader = DataLoader(tokenized_train_datasets, batch_size=batch_size)

        # set training mode
        generator2_train.train()
        discriminator_cnn.train()
        # update_learning_rate(num_update, 8e4, args.g_learning_rate, args.lr_shrink, g_optimizer)

        # Initialize loss for this epoch # Usually, the losses for G and D are not calculated during the training phase, but just the model parameters are updated. However, we are just capturing these metrics for analysis purposes.

        total_train_g_loss = 0
        total_train_d_loss = 0

        for i, sample in enumerate(train_dataloader):

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # sample = sample.to(device)
            # Move all tensors in the batch to the device
            sample = {key: tensor.to(device) for key, tensor in sample.items()}

            # Get the source and target sentences from the batch
            src_sentences = sample["input_ids"]
            tgt_sentences = sample["target_ids"]

            # Train the generator
            optimizer_g.zero_grad()
            fake_tgt_sentences_probs = generator2_train(src_sentences, tgt_sentences)
            
            fake_tgt_sentences_probs = fake_tgt_sentences_probs.view(-1, fake_tgt_sentences_probs.size(-1))  # Shape: [batch_size * seq_len, vocab_size]
            tgt_sentences_flat = tgt_sentences.view(-1)  # Shape: [batch_size * seq_len]

            
            g_loss = g_criterion(fake_tgt_sentences_probs, tgt_sentences_flat)
            g_loss.backward()
            optimizer_g.step()
            total_train_g_loss += g_loss.item()

            # Train the discriminator
            optimizer_d.zero_grad()
            print("src_sentences shape ", src_sentences.shape)
            print("tgt_sentences shape ", tgt_sentences.shape)
            
            # Ensure targets for real and fake are correctly shaped
            real_targets = torch.ones(src_sentences.size(0), 1).to(device)  # Real
            fake_targets = torch.zeros(src_sentences.size(0), 1).to(device)  # Fake
            # disc_out_humanTranSent = discriminator_cnn(src_sentences, tgt_sentences)
            
            real_loss = d_criterion(
                discriminator_cnn(src_sentences, tgt_sentences),
                real_targets,
            )
            
            # preparing the fake sentence probs output from the generator to feed to the discriminator
            print("fake_tgt_sentences_probs shape ", fake_tgt_sentences_probs.shape)
            _, prediction = fake_tgt_sentences_probs.topk(1)
            print("prediction shape ", prediction.shape)
            prediction = prediction.squeeze(1)
            print("prediction shape after squeeze ", prediction.shape)
            fake_tgt_sentences = torch.reshape(prediction, src_sentences.shape)
            print("fake_tgt_sentences shape ", fake_tgt_sentences.shape)
            print("src_sentences shape ", src_sentences.shape)
            
            fake_loss = d_criterion(
                discriminator_cnn(src_sentences, fake_tgt_sentences.detach()),
                fake_targets,
            )
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_d.step()
            total_train_d_loss += d_loss.item()
        
    # Print Training losses
    print(f"Training Generator Loss: {total_train_g_loss / len(train_dataloader)}")
    print(f"Training Discriminator Loss: {total_train_d_loss / len(train_dataloader)}")
    
    torch.save({
        'epoch': epoch_i,
        'generator_state_dict': generator2_train.state_dict(),
        'discriminator_state_dict': discriminator_cnn.state_dict(),
        'optimizer_g_state_dict': optimizer_g.state_dict(),
        'optimizer_d_state_dict': optimizer_d.state_dict(),
        'g_loss': g_loss,
        'd_loss': d_loss,
    }, checkpoints_path + f'train_checkpoint_{epoch_i}.pt')


    #### ----------------------------------VALIDATION --------------------#####

    # After training, switch to evaluation mode
    generator2_train.eval()
    discriminator_cnn.eval()

    # Initialize loss for this epoch
    total_valid_g_loss = 0
    total_valid_d_loss = 0

    # Creating dataloader for validation
    tokenized_valid_datasets.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "target_ids", "target_attention_mask"],
    )
    valid_dataloader = DataLoader(tokenized_valid_datasets, batch_size=batch_size)

    with torch.no_grad():
        for i, sample in enumerate(valid_dataloader):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            sample = sample.to(device)

            # Get the source and target sentences from the batch
            src_sentences = sample["input_ids"]
            tgt_sentences = sample["target_ids"]

            # Generate sentences
            fake_tgt_sentences = generator2_train(src_sentences)

            # Calculate generator loss
            g_loss = g_criterion(fake_tgt_sentences, tgt_sentences)
            total_valid_g_loss += g_loss.item()

            # Calculate discriminator loss
            real_loss = d_criterion(
                discriminator_cnn(src_sentences, tgt_sentences),
                torch.ones_like(tgt_sentences),
            )
            fake_loss = d_criterion(
                discriminator_cnn(src_sentences, fake_tgt_sentences),
                torch.zeros_like(fake_tgt_sentences),
            )
            d_loss = (real_loss + fake_loss) / 2
            total_valid_d_loss += d_loss.item()

    # Print validation losses
    print(f"Validation Generator Loss: {total_valid_g_loss / len(valid_dataloader)}")
    print(f"Validation Discriminator Loss: {total_valid_d_loss / len(valid_dataloader)}")
    
    # After each epoch of validation, save the model and optimizer states
    torch.save({
        'epoch': epoch_i,
        'generator_state_dict': generator2_train.state_dict(),
        'discriminator_state_dict': discriminator_cnn.state_dict(),
        'optimizer_g_state_dict': optimizer_g.state_dict(),
        'optimizer_d_state_dict': optimizer_d.state_dict(),
        'g_loss': total_valid_g_loss / len(valid_dataloader),
        'd_loss': total_valid_d_loss / len(valid_dataloader),
    }, f'validation_checkpoint_{epoch_i}.pt')

    # Save the model and optimizer states in pickle files only when the validation loss is less than the best loss
    
    # Calculate average validation loss
    avg_val_loss = (total_valid_g_loss + total_valid_d_loss) / (2 * len(valid_dataloader))

    # If the validation loss improved, save the entire model
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(generator2_train, open(f'best_generator.pt', 'wb'), pickle_module=dill)
        torch.save(discriminator_cnn, open(f'best_discriminator.pt', 'wb'), pickle_module=dill)
        
if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        logging.warning(f"unknown arguments: {parser.parse_known_args()[1]}")
    main(options)
