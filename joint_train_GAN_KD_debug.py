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
etpwd = os.getcwd()
# sys.path.append(
#     "/u/prattisr/phase-2/all_repos/Adversarial_NMT/neural-machine-translation-using-gan-master"
# )
sys.path.append(getpwd)
# https://stackoverflow.com/questions/67311527/how-to-set-gpu-count-to-0-using-os-environcuda-visible-devices
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
"""
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
torch.cuda.device_count() # result is 2

os.environ["CUDA_VISIBLE_DEVICES"]="0"
torch.cuda.device_count() # result is 1, using first GPU

os.environ["CUDA_VISIBLE_DEVICES"]="1"
torch.cuda.device_count() # result is 1, using second GPU"""
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


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

g_and_d_loss_checkpoint_config =[
        { "combination" : "G_0_0_0_0_1_cos_kl_pg_rkldlgts_0_0_0_D_x_to_1_mil_Bias_T_LM_T_PGloss_1_2_upd_bs_40_0PG_100rkld_lgts_00001lr_ep_5",
    "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":00.00,"g_kl_loss":0.00, "g_pg_loss":0, "g_rkld_logits":100}, 
    "d_loss" : {"real_loss":0.0, "fake_loss":0.0, "fake_loss_pretrain":0.0},
    "gradient_update": {"BIAS": True, "LM": False},
    "Dataset":{"train_size":1000},
    "Mscll":{"Comments": "Baseline : This is the combination with Only Bias layers updating and LM layer freezed"}
    }
]

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

    # Loading data using datasets library from the HuggingFace
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
    # train_dataset = dataset["train"].select(range(1000000))
    # train_dataset = dataset["train"].select(range(100000))
    # train_dataset = dataset["train"].select(range(1000))
    train_dataset = dataset["train"].select(range(config['Dataset']['train_size']))

    # Keep the full valid and test datasets
    valid_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    # Loading Bert Model
    # bert_model = "bert-base-multilingual-cased"

    # Pre-processing the data
    # To-Do : Need to change the max_length to 50 from 128
    source_lang = "en"
    target_lang = "fr"
    prefix = ""

    from transformers import AutoTokenizer

    checkpoint = 'Helsinki-NLP/opus-mt-en-fr'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def preprocess_MarianMT(examples):
        # Initialize the BERT tokenizer
        # tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
        

        # checkpoint = "google-t5/t5-small"
        # checkpoint = 'Helsinki-NLP/opus-mt-en-fr'
        # tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        en = list()
        fr = list()
        for element in examples["translation"]:
            # print("element: ", element)
            en.append(element["en"])
            fr.append(element["fr"])
        
        en = [prefix + text for text in en]

        # Tokenize the data
        inputs = tokenizer(en, truncation=True, padding="max_length", max_length=128)
        with tokenizer.as_target_tokenizer():
            targets = tokenizer(fr, truncation=True, padding="max_length", max_length=128)

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

    tokenized_train_datasets = train_dataset.map(preprocess_MarianMT, batched=True)
    tokenized_valid_datasets = valid_dataset.map(preprocess_MarianMT, batched=True)

        #### -------------------Loading G1 using hub.load--------------------#####
    
    generator1_pretrained = torch.hub.load('pytorch/fairseq', 'transformer.wmt14.en-fr', tokenizer='moses', bpe='subword_nmt')

    # Specify the path to your dictionary file
    
    pwd = os.getcwd()
    if not os.path.exists(os.path.join(pwd, "pretrained_models", "wmt14.en-fr.joined-dict.transformer")):
        os.makedirs(os.path.join(pwd, "pretrained_models", "wmt14.en-fr.joined-dict.transformer"))
    
    print("G1 - Pre-Trained fairseq Generator loaded successfully using hub.load !")
    
    # G1 - Pre-Trained fairseq Generator : Help methods #
    def ids_to_sentences_bert(input_ids):
        """
        Converts lists of token IDs back into sentences using the BERT tokenizer.

        Args:
            input_ids (list[list[int]]): A list of lists containing token IDs.
            tokenizer (BertTokenizerFast): An instance of BertTokenizerFast used for decoding.

        Returns:
            list[str]: A list of decoded sentences.
        """
    
        sentences = []

        # Convert each list of token IDs back into a sentence
        for ids in input_ids:
            # Decode the token IDs to a sentence, skipping special tokens
            sentence = tokenizer.decode(ids, skip_special_tokens=True)
            sentence = sentence.replace("▁", " ").strip()
            sentences.append(sentence)

        return sentences

    def sentences_to_ids(sentences, max_length=128):
        """
        Tokenizes sentences and returns their corresponding input IDs and attention masks,
        suitable for input into a BERT-based discriminator.

        Args:
            sentences (list[str]): A list of sentences to tokenize.
            tokenizer (BertTokenizerFast): An instance of BertTokenizerFast for tokenization.
            max_length (int): Maximum sequence length.

        Returns:
            dict: A dictionary containing 'input_ids' and 'attention_mask' for the tokenized sentences.
        """
        encoding = tokenizer(
            sentences,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }
    
    #### --------------------Loading G2 - Generator in Train() in GAN--------------------#####

    from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
    generator2_train = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    #### --------------------Loading D - Discriminator in Train() in GAN --------------------#####

    discriminator_cnn = Discriminator_cnn_bert(args, use_cuda=use_cuda)

    ##### ---- Other set-up before starting JOINT TRANING-------############

    if use_cuda:
        if torch.cuda.device_count() > 1:
            discriminator_cnn = torch.nn.DataParallel(discriminator_cnn).cuda()
            generator2_train = torch.nn.DataParallel(generator2_train).cuda()
            generator1_pretrained = torch.nn.DataParallel(generator1_pretrained).cuda()
        else:
            generator2_train.cuda()
            discriminator_cnn.cuda()
            generator1_pretrained.cuda()
    else:
        discriminator_cnn.cpu()
        generator2_train.cpu()
        generator1_pretrained.cpu()
 
    # G_0_0_0_0_1_cos_kl_pg_rkldlgts_0_0_0_D_x_to_1_mil_Bias_F_LM_T_PGloss_1_2_upd_bs_40_0PG_100rkld_lgts_00001lr_ep_5
    if not os.path.exists("checkpoints/bert_dualG/wmt14_en_fr_1mil_"+config['combination']+"_"+str(config['Dataset']['train_size'])+"_save_open_direct_pretrained"):
        os.makedirs("checkpoints/bert_dualG/wmt14_en_fr_1mil_"+config['combination']+"_"+str(config['Dataset']['train_size'])+"_save_open_direct_pretrained")
    checkpoints_path = "checkpoints/bert_dualG/wmt14_en_fr_1mil_"+config['combination']+"_"+str(config['Dataset']['train_size'])+"_save_open_direct_pretrained/"

    # Define the policy gradient loss function
    def policy_gradient_loss(discriminator, src_sentences, fake_tgt_sentences,rewards):
        """
        Calculate the policy gradient loss for the generator, aligning with the Discriminator_cnn_bert's requirements.
        
        Args:
            discriminator: The Discriminator_cnn_bert model.
            src_sentences: Tensor of real source sentences.
            fake_tgt_sentences: Tensor of generated target sentences by the generator.
            rewards: Tensor of rewards from the discriminator for each generated sentence pair.
        
        Returns:
            loss: The computed policy gradient loss.
        """
        # Here we call the discriminator with both the source and fake target sentences
        # It's assumed that rewards are the discriminator's output for these pairs
        discriminator_scores = discriminator(src_sentences, fake_tgt_sentences).squeeze()
        print("type of discriminator_scores ", type(discriminator_scores))
        print("shape of discriminator_scores ", discriminator_scores.shape)
        
        # Assuming the discriminator_scores are probabilities (after sigmoid in the discriminator),
        # directly use them for calculating the loss. If they're logits, apply sigmoid here.
        loss = -torch.mean(rewards * torch.log(discriminator_scores + 1e-8))
        print("type of loss ", type(loss))
        print("loss ", loss)
        
        return loss


    ## Define loss functions for the generator and the Discriminator
    g_criterion = nn.CrossEntropyLoss(reduction="sum", ignore_index=0)
    d_criterion = torch.nn.BCELoss()

    #### ----------------------------------JOINT TRAINING --------------------#####

    # Define the optimizers
    optimizer_g = torch.optim.Adam(
    filter(lambda p: p.requires_grad, generator2_train.parameters()), 
    lr=0.00001
    )
    optimizer_d = torch.optim.Adam(discriminator_cnn.parameters(), lr=0.001)

    
    # Start the training loop
    import os

    def remove_db_if_exists(db_path):
        """
        Checks if a SQLite database file exists at the specified path and removes it if it does.

        Args:
        db_path (str): The file path to the SQLite database.
        """
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"Database at '{db_path}' has been removed.")
        else:
            print(f"No database found at '{db_path}' to remove.")

    # Example usage
    getpwd = os.getcwd()
    # db_name = "translations_600sents_debug_spchars_v2.db"
    db_name = "translations_wmt14_en_fr_1mil_pg_kd_loss_MarianMT_unfreezeonlylmlayer_1mil_1epochs_comb_"+config['combination']+"save_pretrained_with_tokenizer_dict_format.db"
    db_path = getpwd + "/" + db_name
    remove_db_if_exists(db_path)
    
    # Early stopping parameters
    # best_val_loss = float('inf')
    best_loss = math.inf
    patience_counter = 0
    patience_threshold = 2  # Example value, adjust as needed

    def cosine_embedding_loss(fake_tgt_sentences_embeds, fr_decoded_bert_embeds):

        batch_size = fake_tgt_sentences_embeds.size(0)
        seq_length = fake_tgt_sentences_embeds.size(1)

        # Create a target tensor for batch
        target = torch.ones((batch_size, seq_length), dtype=torch.float, device=fake_tgt_sentences_embeds.device)

        cosine_loss = torch.nn.functional.cosine_embedding_loss(
        fake_tgt_sentences_embeds.view(-1, fake_tgt_sentences_embeds.size(2)),  # Reshape to [batch_size * seq_length, embedding_size]
        fr_decoded_bert_embeds.view(-1, fr_decoded_bert_embeds.size(2)),  # Reshape similarly
        target.view(-1),  # Flatten the target to match the reshaped embeddings
        margin=0.5)

        return cosine_loss
    
    def kl_divergence_loss(fake_tgt_sentences_embeds, fr_decoded_bert_embeds):

        # Normalize embeddings to probability distributions
        prob_fairseq = torch.nn.functional.softmax(fr_decoded_bert_embeds, dim=-1)
        prob_marian = torch.nn.functional.softmax(fake_tgt_sentences_embeds, dim=-1)

        # Compute KL divergence from Marian to Fairseq
        kl_divergence = torch.nn.functional.kl_div(torch.log(prob_marian), prob_fairseq, reduction='batchmean')
        print("KL Divergence (Marian || Fairseq):", kl_divergence.item())

        return kl_divergence

    def kl_divergence_loss_reverse(fake_tgt_sentences_embeds, fr_decoded_bert_embeds):

        # Normalize embeddings to probability distributions
        prob_fairseq = torch.nn.functional.softmax(fr_decoded_bert_embeds, dim=-1)
        prob_marian = torch.nn.functional.softmax(fake_tgt_sentences_embeds, dim=-1)

        # Compute KL divergence from Fairseq to Marian
        kl_divergence_reverse = torch.nn.functional.kl_div(torch.log(prob_fairseq), prob_marian, reduction='batchmean')
        print("KL Divergence (Fairseq || Marian):", kl_divergence_reverse.item())

        return kl_divergence_reverse
    
    def extract_fairseq_logits(fairseq_hub, train_dataset, src_sentences_for_G1):
        fairseq_model = fairseq_hub.models[0]
        source_dictionary = fairseq_hub.task.source_dictionary

        # Get the device of the model
        device = next(fairseq_model.parameters()).device
        
        fairseq_model.eval()
        fairseq_logits_list = []
        with torch.no_grad():
            for raw_text in src_sentences_for_G1:
                # Encode using Fairseq source dictionary
                tokens = source_dictionary.encode_line(raw_text, add_if_not_exist=False).long().unsqueeze(0)  # Add batch dimension
                tokens = tokens.to(device)
                src_lengths = torch.tensor([tokens.size(1)], device=device)  # Sequence length

                encoder_out = fairseq_model.encoder(tokens, src_lengths=src_lengths)
                bos_token = fairseq_hub.task.target_dictionary.bos()
                prev_output_tokens = torch.full((1, 1), bos_token, dtype=torch.long, device=device)
                
                # Autoregressively generate logits for each token
                logits = []
                for _ in range(tokens.size(1)):
                    decoder_output = fairseq_model.decoder(prev_output_tokens, encoder_out=encoder_out)
                    # Get the logits for the last predicted token
                    last_token_logits = F.log_softmax(decoder_output[0][:, -1, :], dim=-1)
                    logits.append(last_token_logits.unsqueeze(1))
                    # Append the most probable token to prev_output_tokens
                    next_token = last_token_logits.argmax(dim=-1, keepdim=True)
                    prev_output_tokens = torch.cat([prev_output_tokens, next_token], dim=1)

                
                
                fairseq_logits_list.append(torch.cat(logits, dim=1))  # Shape: (1, seq_len, vocab_size)
                
        return fairseq_logits_list

    def reverse_kld_logits(generator_teacher_logits_list, generator_student_logits_list, fs_vocab_size, marian_vocab_size, max_length=128):
        reverse_kld = []
        for generator_teacher_logits, generator_student_logits in zip(generator_teacher_logits_list, generator_student_logits_list):
            print(f"generator_teacher_logits shape: {generator_teacher_logits.shape}")
            print(f"generator_student_logits shape: {generator_student_logits.shape}")
            
            if len(generator_student_logits.shape)==2 :
                generator_student_logits = generator_student_logits.unsqueeze(0)
            
            # Adjust to the same vocabulary size
            if fs_vocab_size < marian_vocab_size:
                generator_teacher_logits = torch.nn.functional.pad(generator_teacher_logits, (0, marian_vocab_size - fs_vocab_size), value=0.0)
            elif fs_vocab_size > marian_vocab_size:
                generator_student_logits = torch.nn.functional.pad(generator_student_logits, (0, fs_vocab_size - marian_vocab_size), value=0.0)
            
            # Pad generator_teacher_logits to the student's max sequence length
            teacher_seq_length = generator_teacher_logits.size(1)
            if teacher_seq_length < max_length:
                generator_teacher_logits = F.pad(generator_teacher_logits, (0, 0, 0, max_length - teacher_seq_length), value=0.0)
            
            # Truncate generator_student_logits to the teacher's sequence length
            student_seq_length = generator_student_logits.size(1)
            if student_seq_length > max_length:
                generator_student_logits = generator_student_logits[:, :max_length, :]
            
            r_kld = torch.sum(generator_student_logits * (generator_student_logits - generator_teacher_logits), dim=-1).mean()
            reverse_kld.append(r_kld)
            
            print(f"After: generator_teacher_logits shape: {generator_teacher_logits.shape}")
            print(f"After: generator_student_logits shape: {generator_student_logits.shape}")
            
            # Convert list of tensors to a single tensor
            reverse_kld_tensor = torch.stack(reverse_kld)
            print(f"reverse_kld_tensor shape: {reverse_kld_tensor.shape}")
            
        return reverse_kld_tensor
    
        

if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        logging.warning(f"unknown arguments: {parser.parse_known_args()[1]}")
    for config in tqdm(g_and_d_loss_checkpoint_config, desc="Running models with diff. g_loss and d_loss"):
        print(" running config ", config["combination"])
        
        # Log the current configuration to the seed file
        with open(seed_file_name, "a") as seed_file:
            seed_file.write(f"Running config: {config['combination']}\n")
            seed_file.write("G Loss: " + str(config["total_g_loss"]) + "\n")
            seed_file.write("D Loss: " + str(config["d_loss"]) + "\n")
            seed_file.write("Gradient Update: " + str(config["gradient_update"]) + "\n")
            seed_file.write("Dataset: " + str(config["Dataset"]) + "\n")
            seed_file.write("Miscellaneous: " + str(config["Mscll"]) + "\n")
            seed_file.write(" ************** end of config ************** \n")
        
        main(options, config)

