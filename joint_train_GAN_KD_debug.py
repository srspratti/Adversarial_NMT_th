# Importing required libraries
import torch
from torch import cuda
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
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
import re
import sqlite3
from datetime import datetime
from generator_tf_bert_t5 import TransformerModel_t5
from discriminator_cnn_bert import Discriminator_cnn_bert

# Set up CUDA
torch.cuda.empty_cache()

# Generate a random seed
seed = 88667  # Example seed value
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Dynamically generate a filename with a timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS

pwd = os.getcwd()
if not os.path.exists(os.path.join(pwd, "checkpoints", "bert_dualG")):
    os.makedirs(os.path.join(pwd, "checkpoints", "bert_dualG"))

seed_file_name = os.path.join(pwd, "checkpoints", "bert_dualG") + f"seed_{timestamp}.txt"

with open(seed_file_name, "w") as seed_file:
    seed_file.write(f"Generated Seed: {seed}\n")
    seed_file.write("Configurations run:\n")

# CUDA multiple-GPU configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# Logging configuration
logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)

parser = argparse.ArgumentParser(description="Adversarial-NMT-BERT")

# Add general arguments
options.add_general_args(parser)
options.add_dataset_args(parser)
options.add_distributed_training_args(parser)
options.add_optimization_args(parser)
options.add_checkpoint_args(parser)
options.add_generator_model_args(parser)
options.add_discriminator_model_args(parser)
options.add_generation_args(parser)

# Model and Training Configurations
def main(args, config):
    use_cuda = torch.cuda.is_available()

    args.encoder_embed_dim = 512
    args.encoder_layers = 2
    args.encoder_dropout_out = 0
    args.decoder_embed_dim = 512
    args.encoder_heads = 2
    args.encoder_ffn_embed_dim = 1000
    args.decoder_heads = 2
    args.decoder_ffn_embed_dim = 1000
    args.decoder_layers = 2
    args.decoder_out_embed_dim = 1000
    args.decoder_dropout_out = 0
    args.bidirectional = False

    # Load the dataset
    dataset_name = "wmt14"
    config_name = "fr-en"
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "datasets")

    dataset = load_dataset(dataset_name, config_name, cache_dir=cache_dir)
    train_dataset = dataset["train"].select(range(config['Dataset']['train_size']))
    valid_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    # Load tokenizer
    checkpoint = 'Helsinki-NLP/opus-mt-en-fr'
    tokenizer = BertTokenizerFast.from_pretrained(checkpoint)

    def preprocess_MarianMT(examples):
        en = [example["en"] for example in examples["translation"]]
        fr = [example["fr"] for example in examples["translation"]]

        inputs = tokenizer(en, truncation=True, padding="max_length", max_length=128)
        with tokenizer.as_target_tokenizer():
            targets = tokenizer(fr, truncation=True, padding="max_length", max_length=128)

        return {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "target_ids": targets.input_ids,
            "target_attention_mask": targets.attention_mask,
        }

    tokenized_train_datasets = train_dataset.map(preprocess_MarianMT, batched=True)
    tokenized_valid_datasets = valid_dataset.map(preprocess_MarianMT, batched=True)

    # Initialize models
    generator = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    discriminator = Discriminator_cnn_bert(args, use_cuda=use_cuda)

    # Set up optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=1e-5)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=1e-3)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()

        # Train generator
        for i, sample in enumerate(DataLoader(tokenized_train_datasets, batch_size=32)):
            inputs = sample["input_ids"].to("cuda")
            outputs = generator(inputs)
            loss_g = nn.CrossEntropyLoss()(outputs.logits, sample["target_ids"].to("cuda"))

            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

            print(f"Epoch {epoch}, Step {i}, Generator Loss: {loss_g.item()}")

        # Validation loop
        generator.eval()
        total_loss = 0
        for sample in DataLoader(tokenized_valid_datasets, batch_size=32):
            with torch.no_grad():
                inputs = sample["input_ids"].to("cuda")
                outputs = generator(inputs)
                loss = nn.CrossEntropyLoss()(outputs.logits, sample["target_ids"].to("cuda"))
                total_loss += loss.item()

        print(f"Epoch {epoch}, Validation Loss: {total_loss / len(tokenized_valid_datasets)}")

if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        logging.warning(f"unknown arguments: {parser.parse_known_args()[1]}")
    main(options, g_and_d_loss_checkpoint_config[0])
