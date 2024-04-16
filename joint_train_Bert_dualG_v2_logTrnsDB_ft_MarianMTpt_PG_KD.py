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
from dictionary import Dictionary
import re
import subprocess
import os

# Importing Generator and Discriminator class methods
# from generator_tf_bert import TransformerModel_bert
from generator_tf_bert_t5 import TransformerModel_t5
from discriminator_cnn_bert import Discriminator_cnn_bert

torch.cuda.empty_cache()

# CUDA multiple-GPU configuration

getpwd = os.getcwd()
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

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
    args.encoder_embed_dim = 512 #768  # 1000 # changed to 768 to match the BERT model
    args.encoder_layers = 2  # 4
    args.encoder_dropout_out = 0
    args.decoder_embed_dim = 512 #768  # 1000 #changed to 768 to match the BERT model
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
    train_dataset = dataset["train"].select(range(1000020))
    # train_dataset = dataset["train"].select(range(200))

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

    # print(train_dataset[0])
    # tokenized_datasets = dataset.map(tokenize_function, batched=True) # using the other berttokenizer map function
    tokenized_train_datasets = train_dataset.map(
        preprocess_MarianMT, batched=True
    )  # Using the bertFaSTtOKENIZER MAp function
    tokenized_valid_datasets = valid_dataset.map(
        preprocess_MarianMT, batched=True
    )  # Using the bertFaSTtOKENIZER MAp function
    tokenized_test_datasets = test_dataset.map(
        preprocess_MarianMT, batched=True
    )  # Using the bertFaSTtOKENIZER MAp function

    #### --------------------Loading G1 - Pre-Trained fairseq Generator --------------------#####
    # path_to_your_pretrained_model = '/root/Adversarial_NMT_th/pretrained_models/wmt14.en-fr.joined-dict.transformer'

    from fairseq.models.transformer import TransformerModel

    getpwd = os.getcwd()
    path_to_your_pretrained_model = (
        getpwd + "/pretrained_models/wmt14.en-fr.joined-dict.transformer"
    )
    generator1_pretrained = TransformerModel.from_pretrained(
        path_to_your_pretrained_model,
        checkpoint_file="model.pt",
        bpe="subword_nmt",
        # data_name_or_path='/u/prattisr/phase-2/all_repos/Adversarial_NMT/neural-machine-translation-using-gan-master/data-bin/wmt14_en_fr_raw_sm/50kLines',
        data_name_or_path=getpwd
        + "/pretrained_models/wmt14.en-fr.joined-dict.transformer",
        bpe_codes=getpwd
        + "/pretrained_models/wmt14.en-fr.joined-dict.transformer/bpecodes",
    )
    print("G1 - Pre-Trained fairseq Generator loaded successfully!")

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
        # # tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
        # from transformers import AutoTokenizer

        # # checkpoint = "google-t5/t5-small"
        # # checkpoint = 'sriram-sanjeev9s/T5_base_wmt14_En_Fr_1million'
        # tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        sentences = []

        # Convert each list of token IDs back into a sentence
        for ids in input_ids:
            # Decode the token IDs to a sentence, skipping special tokens
            sentence = tokenizer.decode(ids, skip_special_tokens=True)
            # sentence = tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True)
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
        # Tokenize the sentences
        # tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
        # from transformers import AutoTokenizer

        # # checkpoint = "google-t5/t5-small"
        # # checkpoint = 'sriram-sanjeev9s/T5_base_wmt14_En_Fr_1million'
        # tokenizer = AutoTokenizer.from_pretrained(checkpoint)
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

    # generator2_train = TransformerModel_t5(args, use_cuda=use_cuda)
    # tokenizer =  .from_pretrained(model_name)
    # generator2_train = MarianMTModel.from_pretrained(model_name)
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

    # adversarial training checkpoints saving path
    if not os.path.exists("checkpoints/bert_dualG/wmt14_en_fr_1mil_pg_kd_loss_MarianMT_unfreezeonlylmlayer_1mil_20epochs_save_pretrained_with_tokenizer_dict_format"):
        os.makedirs("checkpoints/bert_dualG/wmt14_en_fr_1mil_pg_kd_loss_MarianMT_unfreezeonlylmlayer_1mil_20epochs_save_pretrained_with_tokenizer_dict_format")
    checkpoints_path = "checkpoints/bert_dualG/wmt14_en_fr_1mil_pg_kd_loss_MarianMT_unfreezeonlylmlayer_1mil_20epochs_save_pretrained_with_tokenizer_dict_format/"

    # if not os.path.exists("checkpoints/bert_dualG/wmt14_en_fr_1mil_pg_kd_loss_MarianMT_unfreezeonlylmlayer_1mil_20epochs_save_pretrained_with_tokenizer"):
    #     os.makedirs("checkpoints/bert_dualG/wmt14_en_fr_1mil_pg_kd_loss_MarianMT_unfreezeonlylmlayer_1mil_20epochs_save_pretrained_with_tokenizer")
    # checkpoints_path = "checkpoints/bert_dualG/wmt14_en_fr_1mil_pg_kd_loss_MarianMT_unfreezeonlylmlayer_1mil_20epochs_save_pretrained_with_tokenizer/"

    # if not os.path.exists("checkpoints/bert_dualG/wmt14_en_fr_1mil_pg_kd_loss_MarianMT_unfreezeonlylmlayer_600sents_dedbug_spcChars__save_pretrained_v2"):
    #     os.makedirs("checkpoints/bert_dualG/wmt14_en_fr_1mil_pg_kd_loss_MarianMT_unfreezeonlylmlayer_600sents_dedbug_spcChars__save_pretrained_v2")
    # checkpoints_path = "checkpoints/bert_dualG/wmt14_en_fr_1mil_pg_kd_loss_MarianMT_unfreezeonlylmlayer_600sents_dedbug_spcChars__save_pretrained_v2/"

    # Definining loss function methods for generator - Additional 

    # Define the policy gradient loss function
    # def policy_gradient_loss(discriminator, src_sentences, fake_tgt_sentences,rewards):
    #     """
    #     Calculate the policy gradient loss for the generator, aligning with the Discriminator_cnn_bert's requirements.
        
    #     Args:
    #         discriminator: The Discriminator_cnn_bert model.
    #         src_sentences: Tensor of real source sentences.
    #         fake_tgt_sentences: Tensor of generated target sentences by the generator.
    #         rewards: Tensor of rewards from the discriminator for each generated sentence pair.
        
    #     Returns:
    #         loss: The computed policy gradient loss.
    #     """
    #     # Here we call the discriminator with both the source and fake target sentences
    #     # It's assumed that rewards are the discriminator's output for these pairs
    #     discriminator_scores = discriminator(src_sentences, fake_tgt_sentences).squeeze()
        
    #     # Assuming the discriminator_scores are probabilities (after sigmoid in the discriminator),
    #     # directly use them for calculating the loss. If they're logits, apply sigmoid here.
    #     loss = -torch.mean(rewards * torch.log(discriminator_scores + 1e-8))
        
    #     return loss

    # Define knowledge distillation loss function
    # from transformers import AutoTokenizer
    # def encode_with_bert(sentences, device):
    #     """
    #     Encode sentences using BERT to obtain embeddings.

    #     Args:
    #         sentences (list of str): The sentences to encode.
    #         tokenizer: The BERT tokenizer.
    #         bert_model: The BERT model.
    #         device: The torch device.

    #     Returns:
    #         torch.Tensor: The BERT embeddings for the sentences.
    #     """
    #     # bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')
    #     # from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, T5EncoderModel

    #     # bert_model = AutoModelForSeq2SeqLM.from_pretrained('sriram-sanjeev9s/T5_wmt14_En_Fr_1million')
    #     # bert_model = T5EncoderModel.from_pretrained('sriram-sanjeev9s/T5_base_wmt14_En_Fr_1million')
    #     bert_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    #     # tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        

    #     # checkpoint = "google-t5/t5-small"
    #     # checkpoint = 'sriram-sanjeev9s/T5_base_wmt14_En_Fr_1million'
    #     # tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        
    #     bert_model = bert_model.to(device)
        
    #     # tokenizer(en, truncation=True, padding="max_length", max_length=128)
    #     encoded_input = tokenizer(sentences, padding="max_length", truncation=True, return_tensors='pt', max_length=128)
    #     # print("shape of encoded_input ", encoded_input.shape)
    #     # print(" type of encoded_input ", type(encoded_input))
    #     input_ids = encoded_input['input_ids'].to(device)
    #     # print("input_ids shape ", input_ids.shape)
    #     attention_mask = encoded_input['attention_mask'].to(device)
    #     # print("attention_mask shape ", attention_mask.shape)

    #     with torch.no_grad():
    #         outputs = bert_model(input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
    #         # print("type outputs ", type(outputs))
    #     embeddings = outputs.decoder_hidden_states
    #     print("type embeddings ", type(embeddings))
    #     print("outputs.decoder_hidden_states shape", embeddings.shape)
    #     return embeddings

    

    # def soft_target_distillation_loss(g2_output, g1_translated_embeddings):
    #     """
    #     Compute a soft target distillation loss between G2's output embeddings and G1's translated embeddings.

    #     Args:
    #         g2_output (torch.Tensor): The output from G2.
    #         g1_translated_embeddings (torch.Tensor): The BERT embeddings of G1's translations.

    #     Returns:
    #         torch.Tensor: The computed loss.
    #     """
    #     # resize_layer = nn.Linear(512, 768)  # Resize from 512 to 768 dimensions
    #     # # Move resize layer to the correct device
    #     # resize_layer = resize_layer.to(device)

    #     print("g2_output shape ", g2_output.shape)
    #     print("g1_translated_embeddings shape ", g1_translated_embeddings.shape)

    #     # Resize G1's embeddings to match G2's output size
    #     # resized_g1_embeddings = resize_layer(g1_translated_embeddings)
    #     # print("g2_output shape ", g2_output.shape)
    #     # print("resized_g1_embeddings shape ", resized_g1_embeddings.shape)

    #     loss = F.mse_loss(g2_output, g1_translated_embeddings)
    #     return loss


    ## Define loss functions for the generator and the Discriminator
    g_criterion = torch.nn.NLLLoss(reduction="sum", ignore_index=0)
    d_criterion = torch.nn.BCELoss()
    pg_criterion = PGLoss(size_average=True, reduce=True)

    #### ----------------------------------JOINT TRAINING --------------------#####

    # Define the optimizers
    # optimizer_g = torch.optim.Adam(generator2_train.parameters(), lr=0.001)
    optimizer_g = torch.optim.Adam(
    filter(lambda p: p.requires_grad, generator2_train.parameters()), 
    lr=0.001
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
    db_name = "translations_wmt14_en_fr_1mil_pg_kd_loss_MarianMT_unfreezeonlylmlayer_1mil_20epochs_save_pretrained_with_tokenizer_dict_format.db"
    db_path = getpwd + "/" + db_name
    remove_db_if_exists(db_path)
    
    # Early stopping parameters
    # best_val_loss = float('inf')
    best_loss = math.inf
    patience_counter = 0
    patience_threshold = 2  # Example value, adjust as needed

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

        # from transformers import DataCollatorForSeq2Seq, AutoTokenizer
        # checkpoint = "sriram-sanjeev9s/T5_wmt14_En_Fr_1million"
        # tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        # train_dataloader = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint, max_length=128)

        # set training mode
        generator2_train.train()
        discriminator_cnn.train()
        generator1_pretrained.eval()
        # update_learning_rate(num_update, 8e4, args.g_learning_rate, args.lr_shrink, g_optimizer)

        # Initialize loss for this epoch # Usually, the losses for G and D are not calculated during the training phase, but just the model parameters are updated. However, we are just capturing these metrics for analysis purposes.

        total_train_g_loss = 0
        total_train_d_loss = 0


        # Freeze embedding layers 
        #Access the underlying MarianMT model from DataParallel wrapper if necessary
        generator2_train = generator2_train.module if hasattr(generator2_train, 'module') else generator2_train

        """
        # Freeze embedding layers
        for param in generator2_train.model.shared.parameters():
            param.requires_grad = False

        for param in generator2_train.model.encoder.embed_tokens.parameters():
            param.requires_grad = False

        for param in generator2_train.model.encoder.embed_positions.parameters():
            param.requires_grad = False

        for param in generator2_train.model.decoder.embed_tokens.parameters():
            param.requires_grad = False

        for param in generator2_train.model.decoder.embed_positions.parameters():
            param.requires_grad = False

        # Freeze early encoder layers (for example, the first 2 layers)
        for layer in generator2_train.model.encoder.layers[:2]:
            for param in layer.parameters():
                param.requires_grad = False
        
        # Similarly, you might consider freezing early decoder layers if needed
        # for layer in generator2_train.model.decoder.layers[:2]:
        #     for param in layer.parameters():
        #         param.requires_grad = False
                
        """
        print("generator2_train ", generator2_train)
        # writer = SummaryWriter()
        # generator2_train.to_text_file("generator2_train.txt")
        for param in generator2_train.model.parameters():
            param.requires_grad = False

        # Unfreeze the lm_head
        for param in generator2_train.lm_head.parameters():
            param.requires_grad = True

        ######-------------------------------------TRAINING --------------------------------------------#####
        for i, sample in enumerate(train_dataloader):

            print("At epoch_i ", epoch_i)
            print("At batch no# ", i)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # sample = sample.to(device)
            # Move all tensors in the batch to the device
            print("device ", device)
            sample = {key: tensor.to(device) for key, tensor in sample.items()}

            # Get the source and target sentences from the batch
            src_sentences = sample["input_ids"]
            tgt_sentences = sample["target_ids"]
            attention_mask = sample["attention_mask"]

            print("src_sentences ", src_sentences.shape)

            # ---------------------------------------------------------Train the generator 2 train()
            optimizer_g.zero_grad()
            # fake_tgt_sentences_probs, decoder_out = generator2_train(src_sentences, tgt_sentences)
            # print("type of fake_tgt_sentences_probs ", type(fake_tgt_sentences_probs))
            # print("fake_tgt_sentences_probs shape ", fake_tgt_sentences_probs.shape)

             # Access the original TransformerModel from the DataParallel wrapper
            generator2_train_dp = generator2_train.module if isinstance(generator2_train, torch.nn.DataParallel) else generator2_train

            generator2_train_out = generator2_train_dp(input_ids=src_sentences, attention_mask=attention_mask, decoder_input_ids=tgt_sentences , output_hidden_states=True, return_dict=True)
            # print("generator2_train_out shape ", generator2_train_out.shape)
            print("type of generator2_train_out", type(generator2_train_out))
            print(" generator2_train_keys() ", generator2_train_out.keys())
            # print("generator2_train_out to_tuple", generator2_train_out.to_tuple())
            # dict_keys(['logits', 'past_key_values', 'decoder_hidden_states', 'encoder_last_hidden_state', 'encoder_hidden_states'])
            print("generator2_train_out logits shape ", generator2_train_out.logits.shape)
            print("generator2_train_out decoder_hidden_states shape ", generator2_train_out.decoder_hidden_states)
            print("generator2_train_out encoder_last_hidden_state shape ", generator2_train_out.encoder_last_hidden_state.shape)
            print("generator2_train_out encoder_hidden_states shape ", generator2_train_out.encoder_hidden_states)
            print("generator2_train_out past_key_values shape ", generator2_train_out.past_key_values)


            """
            output_ids = generator2_train_dp.generate(input_ids=src_sentences, attention_mask=attention_mask, return_dict=True)
            print("output_ids shape ", output_ids.shape)
            print("type of output_ids", type(output_ids))
            print("output_ids ", output_ids)
            """
            
            # translated_texts = [tokenizer_g2.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in output_ids]

            fake_tgt_sentences_probs = F.log_softmax(generator2_train_out.logits, dim=-1)

            fake_tgt_sentences_probs = fake_tgt_sentences_probs.view(
                -1, fake_tgt_sentences_probs.size(-1)
            )  # Shape: [batch_size * seq_len, vocab_size]
            tgt_sentences_flat = tgt_sentences.view(-1)  # Shape: [batch_size * seq_len]
            # print(
            #     "fake_tgt_sentences_probs shape after view",
            #     fake_tgt_sentences_probs.shape,
            # )
            # print("tgt_sentences_flat shape ", tgt_sentences_flat.shape)

            g_loss = g_criterion(fake_tgt_sentences_probs, tgt_sentences_flat)


            # -------------------------------------Generating translations using the pre-trained generator
            src_sentences_for_G1 = ids_to_sentences_bert(src_sentences)
            print("src_sentences_for_G1 ", src_sentences_for_G1)
            print("At epoch_i ", epoch_i)
            print("At batch no# ", i)
            print("generating translations using the pre-trained generator")

            """Just for debugging"""
            # for sentence in src_sentences_for_G1:
            #     # print("sentence no#  ", n)
            #     print(sentence)

             # Access the original TransformerModel from the DataParallel wrapper
            original_generator_pt = generator1_pretrained.module if isinstance(generator1_pretrained, torch.nn.DataParallel) else generator1_pretrained

            # Now use the translate method
            # translations = original_generator_pt.translate(sentences)

            translated_sentences_from_G1 = original_generator_pt.translate(
                src_sentences_for_G1
            )
            # print("translated_sentences_from_G1 ", translated_sentences_from_G1)
            # print("translated_sentences_from_G1 shape ", translated_sentences_from_G1.shape)
            print("sentences translated from G1 ")
            print("At epoch_i ", epoch_i)
            print("At batch no# ", i)
            # Convert the translated sentences back into token IDs and attention masks
            fake_tgt_sentences_G1_pretrain_probs = sentences_to_ids(
                translated_sentences_from_G1, max_length=128
            )

            # print(
            #     " type of fake_tgt_sentences_G1_pretrain_probs ",
            #     type(fake_tgt_sentences_G1_pretrain_probs),
            # )
            # print(
            #     "fake_tgt_sentences_G1_pretrain_probs dict keys ",
            #     fake_tgt_sentences_G1_pretrain_probs.keys(),
            # )
            # print(
            #     "size of dict fake_tgt_sentences_G1_pretrain_probs ",
            #     len(fake_tgt_sentences_G1_pretrain_probs),
            # )
            # print(
            #     "dict info key input_ids fake_tgt_sentences_G1_pretrain_probs ",
            #     fake_tgt_sentences_G1_pretrain_probs.get("input_ids"),
            # )
            # print(
            #     "dict info key input_ids type of fake_tgt_sentences_G1_pretrain_probs ",
            #     type(fake_tgt_sentences_G1_pretrain_probs.get("input_ids")),
            # )
            # print(
            #     " shape of dict keys input_ids fake_tgt_sentences_G1_pretrain_probs ",
            #     fake_tgt_sentences_G1_pretrain_probs.get("input_ids").shape,
            # )

            # Now, encoded_translations["input_ids"] and encoded_translations["attention_mask"]
            # can be fed into the discriminator for further processing.
            # fake_tgt_sentences_G1_pretrain_probs = fake_tgt_sentences_G1_pretrain_probs.view(-1, fake_tgt_sentences_G1_pretrain_probs.size(-1))

            # -------------------------------------------------Train the discriminator
            print("training discriminator")
            print("At epoch_i ", epoch_i)
            print("At batch no# ", i)
            optimizer_d.zero_grad()
            # print("src_sentences shape ", src_sentences.shape)
            # print("tgt_sentences shape ", tgt_sentences.shape)

            # Ensure targets for real and fake are correctly shaped
            real_targets = torch.ones(src_sentences.size(0), 1).to(device)  # Real
            fake_targets = torch.zeros(src_sentences.size(0), 1).to(device)  # Fake
            # disc_out_humanTranSent = discriminator_cnn(src_sentences, tgt_sentences)

            # --------------------------------Real loss of the discriminator --------------
            real_loss = d_criterion(
                discriminator_cnn(src_sentences, tgt_sentences),
                real_targets,
            )

            # -----------------------------------------------------Generator 2 Train()-----------------------------------------
            print("At epoch_i ", epoch_i)
            print("At batch no# ", i)
            print("training generator 2")
            # preparing the fake sentence probs output from the generator to feed to the discriminator
            # print("fake_tgt_sentences_probs shape ", fake_tgt_sentences_probs.shape)
            _, prediction = fake_tgt_sentences_probs.topk(1)
            # print("prediction shape ", prediction.shape)
            prediction = prediction.squeeze(1)
            # print("prediction shape after squeeze ", prediction.shape)
            fake_tgt_sentences = torch.reshape(prediction, src_sentences.shape)
            # print("fake_tgt_sentences shape ", fake_tgt_sentences.shape)
            # print("src_sentences shape ", src_sentences.shape)

            # ----------------------------------------- Generator 1 Pre-Trained ---------------------------------------------
            # preparing the fake sentence probs output from the generator 1 Pre-Trained to feed to the discriminator
            # We don't need the below processing because we are using the sentences_to_ids() method to convert the translated sentences from G1 to token IDs and attention masks
            """
            print("fake_tgt_sentences_G1_pretrain_probs shape ", fake_tgt_sentences_G1_pretrain_probs.shape)
            _, prediction_pretrain = fake_tgt_sentences_G1_pretrain_probs.topk(1)
            print("prediction_pretrain shape ", prediction_pretrain.shape)
            prediction_pretrain = prediction_pretrain.squeeze(1)
            print("prediction_pretrain shape after squeeze ", prediction_pretrain.shape)
            fake_tgt_sentences_G1_pretrain = torch.reshape(prediction_pretrain, src_sentences.shape)
            print("fake_tgt_sentences_G1_pretrain shape ", fake_tgt_sentences_G1_pretrain.shape)
            """
            print("src_sentences shape ", src_sentences.shape)

            ###################################### Modified Generator loss 
            # Including policy gradient loss

            # Optional: Incorporate discriminator feedback directly into G2's loss
            # Assume discriminator provides a reward signal for PG training
            # rewards = discriminator_cnn(src_sentences, fake_tgt_sentences.detach()).detach()
            # pg_loss = policy_gradient_loss(discriminator_cnn, src_sentences, fake_tgt_sentences, rewards)


            # Including soft-Knowledge distillation loss
            
            # g1_translated_embeddings = encode_with_bert(translated_sentences_from_G1, device)   
            # # print("g1_translated_embeddings ", g1_translated_embeddings) 
            # soft_target_loss = soft_target_distillation_loss(decoder_out, g1_translated_embeddings)


            # #################################################   Total G2 loss
            # total_g_loss = g_loss + pg_loss  + soft_target_loss # You can adjust the weighting of these components as needed
            total_g_loss = g_loss

            total_g_loss.backward()
            optimizer_g.step()
            total_train_g_loss += total_g_loss.item() # 

            # ---------------------------------------- fake loss from Generator 2 Train()--------------------------
            fake_tgt_sentences = fake_tgt_sentences.to(device)
            # print("fake_tgt_sentences shape ", fake_tgt_sentences.shape)
            fake_loss = d_criterion(
                discriminator_cnn(src_sentences, fake_tgt_sentences.detach()),
                fake_targets,
            )

            # --------------------------------------- fake loss from Generator 1 Pre-Trained--------------------------
            fake_tgt_sentences_G1_pretrain = fake_tgt_sentences_G1_pretrain_probs.get(
                "input_ids"
            )
            fake_tgt_sentences_G1_pretrain = fake_tgt_sentences_G1_pretrain.to(device)
            fake_loss_pretrain = d_criterion(
                discriminator_cnn(
                    src_sentences, fake_tgt_sentences_G1_pretrain.detach()
                ),
                fake_targets,
            )

            ####---LOGGING TRANSLATIONS --------------------------------####

            print("logging translations train\n")
            print("At epoch_i ", epoch_i)
            print("At batch no# ", i)
            print("src_sentences shape English Source", src_sentences.shape)
            print("tgt_sentences shape French Human", tgt_sentences.shape)
            # print(
            #     "fake_tgt_sentences from Generator 2 Train() shape ",
            #     fake_tgt_sentences.shape,
            # )
            # print(
            #     "fake_tgt_sentences_G1_pretrain shape from Generator 1 Pre-Trained ",
            #     fake_tgt_sentences_G1_pretrain.shape,
            # )

            

            # converting these into sentences from ID's using the BERT tokenizer
            
            print("source sentences before converting: ", src_sentences)
            src_sentences_converted_logging_org = ids_to_sentences_bert(src_sentences)
            print("source sentences After converting: ", src_sentences_converted_logging_org)

            # print("tgt_sentences  before converting: ", tgt_sentences)
            tgt_sentences_converted_logging_org = ids_to_sentences_bert(tgt_sentences)
            # print(" tgt_sentences After converting: ", tgt_sentences_converted_logging_org)
                  

            fake_tgt_sentences_converted_logging_G2_train = ids_to_sentences_bert(
                fake_tgt_sentences
            )

            print("fake_tgt_Sentences_G1_pretrain ", fake_tgt_sentences_G1_pretrain)
            fake_tgt_sentences_G1_pretrain_converted_logging = ids_to_sentences_bert(
                fake_tgt_sentences_G1_pretrain
            )
            print("fake_tgt_sentences_G1_pretrain_converted_logging ", fake_tgt_sentences_G1_pretrain_converted_logging)



            fake_tgt_sentences_G1_pretrain_org_translated_sent = translated_sentences_from_G1
            

            # print("type of src_sentences_converted_logging_org ", type(src_sentences_converted_logging_org))
            # print("type of tgt_sentences_converted_logging_org ", type(tgt_sentences_converted_logging_org))
            # print("type of fake_tgt_sentences_converted_logging_G2_train ", type(fake_tgt_sentences_converted_logging_G2_train))
            # print("type of fake_tgt_sentences_G1_pretrain_converted_logging ", type(fake_tgt_sentences_G1_pretrain_converted_logging))
            # print("type of fake_tgt_sentences_G1_pretrain_org_translated_sent ", type(translated_sentences_from_G1))
            
            
            import sqlite3
            
            def init_db_train():
                try:
                    conn = sqlite3.connect(db_name)
                    c = conn.cursor()    
                    c.execute(
                        """CREATE TABLE IF NOT EXISTS translations_train
                                    (id INTEGER PRIMARY KEY, epoch_i_list INTEGER NOT NULL, src_sentences_converted_logging_org TEXT NOT NULL, tgt_sentences_converted_logging_org TEXT NOT NULL, fake_tgt_sentences_converted_logging_G2_train TEXT NOT NULL, fake_tgt_sentences_G1_pretrain_converted_logging TEXT NOT NULL, fake_tgt_sentences_G1_pretrain_org_translated_sent TEXT NOT NULL)"""
                    )
                    conn.commit() #
                except sqlite3.Error as e:
                    print("sqlite3 error: ", e)
                finally:
                    conn.close()

            print("epoch_i ", epoch_i)

            #
            def clean_text(text):
                import html
                text = html.unescape(text)  # Decode HTML entities
                text = text.replace('@-@', '')  # Remove '@-@'
                return text
            
            def log_translation_db_train(
                epoch_i,
                src_sentences_converted_logging_org,
                tgt_sentences_converted_logging_org,
                fake_tgt_sentences_converted_logging_G2_train,
                fake_tgt_sentences_G1_pretrain_converted_logging,
                fake_tgt_sentences_G1_pretrain_org_translated_sent,
            ):
                try:
                    conn = sqlite3.connect(db_name)
                    c = conn.cursor()
                    size_of_src_sentences_converted_logging_org = len(src_sentences_converted_logging_org)
                    epoch_i_list = [epoch_i] * size_of_src_sentences_converted_logging_org
                    print("epoch_i_list ", epoch_i_list)
                    print("size of epoch_i_list ", len(epoch_i_list))
                    for epoch_i_list, src, tgt, fake_tgt_G2, fake_tgt_G1, fake_tgt_G1_org in zip(
                        epoch_i_list ,src_sentences_converted_logging_org,
                        tgt_sentences_converted_logging_org,
                        fake_tgt_sentences_converted_logging_G2_train,
                        fake_tgt_sentences_G1_pretrain_converted_logging,
                        fake_tgt_sentences_G1_pretrain_org_translated_sent):    
                        c.execute(
                            """INSERT INTO translations_train (epoch_i_list, src_sentences_converted_logging_org, tgt_sentences_converted_logging_org, fake_tgt_sentences_converted_logging_G2_train, fake_tgt_sentences_G1_pretrain_converted_logging, fake_tgt_sentences_G1_pretrain_org_translated_sent)
                                        VALUES (?, ?, ?, ?, ?, ?)""",
                            (
                               epoch_i_list, src, tgt, fake_tgt_G2, fake_tgt_G1, fake_tgt_G1_org
                            ),
                        )
                    conn.commit()
                except sqlite3.Error as e:
                    print("sqlite3 error: ", e)
                finally:
                    conn.close()

            # Executing DB logging statements
            
            init_db_train()  # Initialize the database and table

            # cleaned_list - remove the '@-@' from the sentences and other html entities
            src_sentences_converted_logging_org = list(map(clean_text, src_sentences_converted_logging_org))
            tgt_sentences_converted_logging_org = list(map(clean_text, tgt_sentences_converted_logging_org))
            fake_tgt_sentences_converted_logging_G2_train = list(map(clean_text, fake_tgt_sentences_converted_logging_G2_train))
            fake_tgt_sentences_G1_pretrain_converted_logging = list(map(clean_text, fake_tgt_sentences_G1_pretrain_converted_logging))
            fake_tgt_sentences_G1_pretrain_org_translated_sent = list(map(clean_text, fake_tgt_sentences_G1_pretrain_org_translated_sent))

            log_translation_db_train(
                epoch_i,
                src_sentences_converted_logging_org,
                tgt_sentences_converted_logging_org,
                fake_tgt_sentences_converted_logging_G2_train,
                fake_tgt_sentences_G1_pretrain_converted_logging,
                fake_tgt_sentences_G1_pretrain_org_translated_sent,
            )

            # d_loss = (real_loss + fake_loss) / 2
            # combining the real and fake loss from the two generators
            d_loss = (real_loss + fake_loss + fake_loss_pretrain) / 3

            d_loss.backward()
            optimizer_d.step()
            total_train_d_loss += d_loss.item()

        # Print Training losses
        print(f"Training Generator Loss: {total_train_g_loss / len(train_dataloader)}")
        print(f"Training Discriminator Loss: {total_train_d_loss / len(train_dataloader)}")

        torch.save(
            {
                "epoch": epoch_i,
                "generator_state_dict": generator2_train.state_dict(),
                "discriminator_state_dict": discriminator_cnn.state_dict(),
                "optimizer_g_state_dict": optimizer_g.state_dict(),
                "optimizer_d_state_dict": optimizer_d.state_dict(),
                "g_loss": total_train_g_loss/ len(train_dataloader),
                "d_loss": total_train_d_loss / len(train_dataloader),
            },
            checkpoints_path + f"train_checkpoint_dict_format_at_{epoch_i}.pt",
        )

        # torch.save(
        #         generator2_train,
        #         open(checkpoints_path + f"train_checkpoint__generator{epoch_i}.pt", "wb"),
        #         pickle_module=dill,
        #     )
        torch.save(
                discriminator_cnn,
                open(checkpoints_path + f"train_checkpoint_discriminator_at_{epoch_i}.pt", "wb"),
                pickle_module=dill,
            )
        generator2_train.save_pretrained(checkpoints_path + f"train_checkpoint_generator_save_pretrained_at_{epoch_i}")
        tokenizer.save_pretrained(checkpoints_path + f"train_checkpoint_tokenizer_save_pretrained_at_{epoch_i}")
        
        

        #### -----------------------------------------VALIDATION --------------------------------------------#####
        print(
            ".........................................Validation block......................................................................"
        )
        print("At epoch_i ", epoch_i)
        print("At batch no# ", i)
        # After training, switch to evaluation mode
        generator2_train.eval()
        discriminator_cnn.eval()
        generator1_pretrained.eval()

        # Initialize loss for this epoch
        total_valid_g_loss = 0
        total_valid_d_loss = 0

        # Creating dataloader for validation
        tokenized_valid_datasets.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "target_ids", "target_attention_mask"],
        )
        valid_dataloader = DataLoader(tokenized_valid_datasets, batch_size=batch_size)

        # from transformers import DataCollatorForSeq2Seq
        # checkpoint = "sriram-sanjeev9s/T5_wmt14_En_Fr_1million"
        # tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        # valid_dataloader = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint, max_length=128)

        with torch.no_grad():
            for i, sample in enumerate(valid_dataloader):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                # Move all tensors in the batch to the device
                sample = {key: tensor.to(device) for key, tensor in sample.items()}

                # Get the source and target sentences from the batch
                src_sentences = sample["input_ids"]
                tgt_sentences = sample["target_ids"]
                attention_mask = sample["attention_mask"]

                # Generate sentences
                # fake_tgt_sentences_probs = generator2_train(src_sentences, tgt_sentences)
                # fake_tgt_sentences_probs, decoder_out = generator2_train(src_sentences, tgt_sentences)

                 # Access the original TransformerModel from the DataParallel wrapper
                generator2_train_dp = generator2_train.module if isinstance(generator2_train, torch.nn.DataParallel) else generator2_train

                generator2_train_out = generator2_train_dp(input_ids=src_sentences, attention_mask=attention_mask, decoder_input_ids=tgt_sentences , output_hidden_states=True, return_dict=True)
                # print("generator2_train_out shape ", generator2_train_out.shape)
                print("type of generator2_train_out", type(generator2_train_out))
                print(" generator2_train_keys() ", generator2_train_out.keys())
                # print("generator2_train_out to_tuple", generator2_train_out.to_tuple())
                # dict_keys(['logits', 'past_key_values', 'decoder_hidden_states', 'encoder_last_hidden_state', 'encoder_hidden_states'])
                print("generator2_train_out logits shape ", generator2_train_out.logits.shape)
                print("generator2_train_out decoder_hidden_states shape ", generator2_train_out.decoder_hidden_states)
                print("generator2_train_out encoder_last_hidden_state shape ", generator2_train_out.encoder_last_hidden_state.shape)
                print("generator2_train_out encoder_hidden_states shape ", generator2_train_out.encoder_hidden_states)
                print("generator2_train_out past_key_values shape ", generator2_train_out.past_key_values)

                fake_tgt_sentences_probs = F.log_softmax(generator2_train_out.logits, dim=-1)
                fake_tgt_sentences_probs = fake_tgt_sentences_probs.view(
                    -1, fake_tgt_sentences_probs.size(-1)
                )  # Shape: [batch_size * seq_len, vocab_size]
                tgt_sentences_flat = tgt_sentences.view(-1)  # Shape: [batch_size * seq_len]

                # Calculate generator loss
                g_loss = g_criterion(fake_tgt_sentences_probs, tgt_sentences_flat)

                # -------------------------------------Generating translations using the pre-trained generator
                src_sentences_for_G1 = ids_to_sentences_bert(src_sentences)
                # print("src_sentences_for_G1 ", src_sentences_for_G1)

                """Just for debugging"""
                # for sentence in src_sentences_for_G1:
                #     print("sentence no#  ", n)
                #     # print(sentence)
                
                # Access the original TransformerModel from the DataParallel wrapper
                original_generator_pt = generator1_pretrained.module if isinstance(generator1_pretrained, torch.nn.DataParallel) else generator1_pretrained

                # Now use the translate method
                # translations = original_generator_pt.translate(sentences)

                translated_sentences_from_G1 = original_generator_pt.translate(
                    src_sentences_for_G1
                )

                # translated_sentences_from_G1 = generator1_pretrained.translate(
                #     src_sentences_for_G1
                # )
                # print("translated_sentences_from_G1 ", translated_sentences_from_G1)

                # Convert the translated sentences back into token IDs and attention masks
                fake_tgt_sentences_G1_pretrain_probs = sentences_to_ids(
                    translated_sentences_from_G1, max_length=128
                )

                # fake_tgt_sentences_G1_pretrain_probs = fake_tgt_sentences_G1_pretrain_probs.view(-1, fake_tgt_sentences_G1_pretrain_probs.size(-1))

                # -------------------------------------------------Discriminator Validation ------------------------
                real_targets = torch.ones(src_sentences.size(0), 1).to(device)  # Real
                fake_targets = torch.zeros(src_sentences.size(0), 1).to(device)  # Fake

                # -----------------------------------------Real loss of the discriminator --------------
                real_loss = d_criterion(
                    discriminator_cnn(src_sentences, tgt_sentences),
                    real_targets,
                )

                # preparing the fake sentence probs output to feed to discriminator -------


                # print("fake_tgt_sentences_probs shape ", fake_tgt_sentences_probs.shape)
                _, prediction = fake_tgt_sentences_probs.topk(1)
                # print("prediction shape ", prediction.shape)
                prediction = prediction.squeeze(1)
                # print("prediction shape after squeeze ", prediction.shape)
                fake_tgt_sentences = torch.reshape(prediction, src_sentences.shape)
                # print("fake_tgt_sentences shape ", fake_tgt_sentences.shape)
                # print("src_sentences shape ", src_sentences.shape)

                #---------
                # preparing the fake sentence probs output from the generator 1 Pre-Trained to feed to the discriminator
                # We don't need the below processing because we are using the sentences_to_ids() method to convert the translated sentences from G1 to token IDs and attention masks
                """
                #print("fake_tgt_sentences_G1_pretrain_probs shape ", fake_tgt_sentences_G1_pretrain_probs.shape)
                _, prediction_pretrain = fake_tgt_sentences_G1_pretrain_probs.topk(1)
                #print("prediction_pretrain shape ", prediction_pretrain.shape)
                prediction_pretrain = prediction_pretrain.squeeze(1)
                print("prediction_pretrain shape after squeeze ", prediction_pretrain.shape)
                fake_tgt_sentences_G1_pretrain = torch.reshape(prediction_pretrain, src_sentences.shape)
                print("fake_tgt_sentences_G1_pretrain shape ", fake_tgt_sentences_G1_pretrain.shape)
                """
                print("src_sentences shape ", src_sentences.shape)

                ###################################### Modified Generator loss 
                # Including policy gradient loss

                # Optional: Incorporate discriminator feedback directly into G2's loss
                # Assume discriminator provides a reward signal for PG training
                # rewards = discriminator_cnn(src_sentences, fake_tgt_sentences.detach()).detach()
                # pg_loss = policy_gradient_loss(discriminator_cnn, src_sentences, fake_tgt_sentences, rewards)


                # Including soft-Knowledge distillation loss
                
                # g1_translated_embeddings = encode_with_bert(translated_sentences_from_G1, device)   
                # # print("g1_translated_embeddings ", g1_translated_embeddings) 
                # soft_target_loss = soft_target_distillation_loss(decoder_out, g1_translated_embeddings)

                #-------------------------------
                # total_g_loss = g_loss + pg_loss  + soft_target_loss
                total_g_loss = g_loss
                
                total_valid_g_loss += total_g_loss.item()

                # ---------------------------------------fake loss from the Generator 2 now in eval() mode --------------------------
                fake_tgt_sentences = fake_tgt_sentences.to(device)
                fake_loss = d_criterion(
                    discriminator_cnn(src_sentences, fake_tgt_sentences.detach()),
                    fake_targets,
                )

                # ---------------------------------------------fake loss from Generator 1 Pre-Trained--------------------------
                fake_tgt_sentences_G1_pretrain = fake_tgt_sentences_G1_pretrain_probs.get(
                    "input_ids"
                )
                fake_tgt_sentences_G1_pretrain = fake_tgt_sentences_G1_pretrain.to(device)
                fake_loss_pretrain = d_criterion(
                    discriminator_cnn(
                        src_sentences, fake_tgt_sentences_G1_pretrain.detach()
                    ),
                    fake_targets,
                )

                #### logging validation translations into DB ----------############
                print("logging translations valid\n")
                print("At epoch_i ", epoch_i)
                print("At batch no# ", i)
                print("src_sentences shape English Source", src_sentences.shape)
                print("tgt_sentences shape French Human", tgt_sentences.shape)
                # print(
                #     "fake_tgt_sentences from Generator 2 Train() shape ",
                #     fake_tgt_sentences.shape,
                # )
                # print(
                #     "fake_tgt_sentences_G1_pretrain shape from Generator 1 Pre-Trained ",
                #     fake_tgt_sentences_G1_pretrain.shape,
                # )

                # converting these into sentences from ID's using the BERT tokenizer
                src_sentences_converted_logging_org = ids_to_sentences_bert(src_sentences)
                tgt_sentences_converted_logging_org = ids_to_sentences_bert(tgt_sentences)
                fake_tgt_sentences_converted_logging_G2_train = ids_to_sentences_bert(
                    fake_tgt_sentences
                )
                fake_tgt_sentences_G1_pretrain_converted_logging = ids_to_sentences_bert(
                    fake_tgt_sentences_G1_pretrain
                )
                fake_tgt_sentences_G1_pretrain_org_translated_sent = translated_sentences_from_G1
                

                # print("type of src_sentences_converted_logging_org ", type(src_sentences_converted_logging_org))
                # print("type of tgt_sentences_converted_logging_org ", type(tgt_sentences_converted_logging_org))
                # print("type of fake_tgt_sentences_converted_logging_G2_train ", type(fake_tgt_sentences_converted_logging_G2_train))
                print("fake_tgt_sentences_converted_logging_G2_train ", fake_tgt_sentences_converted_logging_G2_train)
                # print("type of fake_tgt_sentences_G1_pretrain_converted_logging ", type(fake_tgt_sentences_G1_pretrain_converted_logging))
                # print("type of fake_tgt_sentences_G1_pretrain_org_translated_sent ", type(translated_sentences_from_G1))
                
                def init_db_valid():
                    try:
                        conn = sqlite3.connect(db_name)
                        c = conn.cursor()    
                        c.execute(
                            """CREATE TABLE IF NOT EXISTS translations_valid
                                        (id INTEGER PRIMARY KEY, epoch_i_list INTEGER NOT NULL, src_sentences_converted_logging_org TEXT NOT NULL, tgt_sentences_converted_logging_org TEXT NOT NULL, fake_tgt_sentences_converted_logging_G2_train TEXT NOT NULL, fake_tgt_sentences_G1_pretrain_converted_logging TEXT NOT NULL, fake_tgt_sentences_G1_pretrain_org_translated_sent TEXT NOT NULL)"""
                        )
                        conn.commit()
                    except sqlite3.Error as e:
                        print("sqlite3 error: ", e)
                    finally:
                        conn.close()

                def log_translation_db_valid(
                    epoch_i,
                    src_sentences_converted_logging_org,
                    tgt_sentences_converted_logging_org,
                    fake_tgt_sentences_converted_logging_G2_train,
                    fake_tgt_sentences_G1_pretrain_converted_logging,
                    fake_tgt_sentences_G1_pretrain_org_translated_sent,
                ):
                    try:
                        conn = sqlite3.connect(db_name)
                        c = conn.cursor()
                        size_of_src_sentences_converted_logging_org = len(src_sentences_converted_logging_org)
                        epoch_i_list = [epoch_i] * size_of_src_sentences_converted_logging_org
                        print("epoch_i_list ", epoch_i_list)
                        for epoch_i_list, src, tgt, fake_tgt_G2, fake_tgt_G1, fake_tgt_G1_org in zip(
                            epoch_i_list, src_sentences_converted_logging_org,
                            tgt_sentences_converted_logging_org,
                            fake_tgt_sentences_converted_logging_G2_train,
                            fake_tgt_sentences_G1_pretrain_converted_logging,
                            fake_tgt_sentences_G1_pretrain_org_translated_sent):    
                            c.execute(
                                """INSERT INTO translations_valid (epoch_i_list, src_sentences_converted_logging_org, tgt_sentences_converted_logging_org, fake_tgt_sentences_converted_logging_G2_train, fake_tgt_sentences_G1_pretrain_converted_logging, fake_tgt_sentences_G1_pretrain_org_translated_sent)
                                            VALUES (?, ?, ?, ?, ?, ?)""",
                                (
                                epoch_i_list, src, tgt, fake_tgt_G2, fake_tgt_G1, fake_tgt_G1_org
                                ),
                            )
                        conn.commit()
                    except sqlite3.Error as e:
                        print("sqlite3 error: ", e)
                    finally:
                        conn.close()

                # Executing DB logging statements
                
                init_db_valid()  # Initialize the database and table

                # cleaned_list - remove the '@-@' from the sentences and other html entities
                src_sentences_converted_logging_org = list(map(clean_text, src_sentences_converted_logging_org))
                tgt_sentences_converted_logging_org = list(map(clean_text, tgt_sentences_converted_logging_org))
                fake_tgt_sentences_converted_logging_G2_train = list(map(clean_text, fake_tgt_sentences_converted_logging_G2_train))
                fake_tgt_sentences_G1_pretrain_converted_logging = list(map(clean_text, fake_tgt_sentences_G1_pretrain_converted_logging))
                fake_tgt_sentences_G1_pretrain_org_translated_sent = list(map(clean_text, fake_tgt_sentences_G1_pretrain_org_translated_sent))

                log_translation_db_valid(
                    epoch_i,
                    src_sentences_converted_logging_org,
                    tgt_sentences_converted_logging_org,
                    fake_tgt_sentences_converted_logging_G2_train,
                    fake_tgt_sentences_G1_pretrain_converted_logging,
                    fake_tgt_sentences_G1_pretrain_org_translated_sent,
                )
                
                ######---------------------------------############################
                
                # combining the real and fake loss from the two generators
                d_loss = (real_loss + fake_loss + fake_loss_pretrain) / 3
                total_valid_d_loss += d_loss.item()

        # Print validation losses
        print(f"Validation Generator Loss: {total_valid_g_loss / len(valid_dataloader)}")
        print(
            f"Validation Discriminator Loss: {total_valid_d_loss / len(valid_dataloader)}"
        )

        # After each epoch of validation, save the model and optimizer states
        torch.save(
            {
                "epoch": epoch_i,
                "generator_state_dict": generator2_train.state_dict(),
                "discriminator_state_dict": discriminator_cnn.state_dict(),
                "optimizer_g_state_dict": optimizer_g.state_dict(),
                "optimizer_d_state_dict": optimizer_d.state_dict(),
                "g_loss": total_valid_g_loss / len(valid_dataloader),
                "d_loss": total_valid_d_loss / len(valid_dataloader),
            },
            checkpoints_path + f"validation_checkpoint_dict_format_at_{epoch_i}.pt",
        )

        # torch.save(
        #         generator2_train,
        #         open(checkpoints_path + f"valid_checkpoint__generator{epoch_i}.pt", "wb"),
        #         pickle_module=dill,
        # #     )
        # torch.save(
        #         discriminator_cnn,
        #         open(checkpoints_path + f"valid_checkpoint_discriminator_at_{epoch_i}.pt", "wb"),
        #         pickle_module=dill,
        #     )
        torch.save(discriminator_cnn, checkpoints_path + f"valid_checkpoint_discriminator_at_{epoch_i}.pt", pickle_module=dill)

        generator2_train.save_pretrained(checkpoints_path + f"valid_checkpoint_generator_save_pretrained_at_{epoch_i}")
        tokenizer.save_pretrained(checkpoints_path + f"valid_checkpoint_tokenizer_save_pretrained_at_{epoch_i}")
        


        # Save the model and optimizer states in pickle files only when the validation loss is less than the best loss

        # Calculate average validation loss
        avg_val_loss = (total_valid_g_loss + total_valid_d_loss) / (
            2 * len(valid_dataloader)
        )

        # If the validation loss improved, save the entire model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            # torch.save(
            #     generator2_train,
            #     open(checkpoints_path + f"best_generator_at_{epoch_i}.pt", "wb"),
            #     pickle_module=dill,
            # )
            # Saving G and Tokenizer using save_pretrained
            generator2_train.save_pretrained(checkpoints_path + f"best_generator_save_pretrained_at_{epoch_i}")
            tokenizer.save_pretrained(checkpoints_path + f"best_generator_tokenizer_save_pretrained_at_{epoch_i}")
        

            # #Saving Discriminator using torch-save()
            # torch.save(
            #     discriminator_cnn,
            #     open(checkpoints_path + f"best_discriminator_at_{epoch_i}.pt", "wb"),
            #     pickle_module=dill,
            # )
            torch.save(
            {
                "epoch": epoch_i,
                "generator_state_dict": generator2_train.state_dict(),
                "discriminator_state_dict": discriminator_cnn.state_dict(),
                "optimizer_g_state_dict": optimizer_g.state_dict(),
                "optimizer_d_state_dict": optimizer_d.state_dict(),
                "g_loss": total_valid_g_loss / len(valid_dataloader),
                "d_loss": total_valid_d_loss / len(valid_dataloader),
            },
            checkpoints_path + f"best_checkpoint_dict_format_{epoch_i}.pt",
            )
            torch.save(discriminator_cnn, checkpoints_path + f"best_discriminator_at_{epoch_i}.pt", pickle_module=dill)
        else:
            patience_counter += 1
        
        if patience_counter >= patience_threshold:
            print(f"Early stopping at epoch {epoch_i}")
            break


if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        logging.warning(f"unknown arguments: {parser.parse_known_args()[1]}")
    main(options)
