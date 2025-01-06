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
# import torch
# from fairseq.data.data_utils import collate_tokens

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
# import data
import utils
from meters import AverageMeter
from PGLoss import PGLoss
from tqdm import tqdm
import re
import subprocess
import os
from datetime import datetime
from generator_tf_bert_t5 import TransformerModel_t5
from discriminator_cnn_bert import Discriminator_cnn_bert

from peft import get_peft_model, LoraConfig, TaskType
# from t3vis.monitor import TrainingMonitor  # T3-Vis Monitoring

torch.cuda.empty_cache()

# Generate a random seed
# seed = random.randint(0, 999999)
seed = 88667

# Set the seed for reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Dynamically generate a filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS

# T3-Vis Training Monitor
# monitor = TrainingMonitor(log_interval=100, output_dir="./t3vis_logs")

pwd = os.getcwd()
if not os.path.exists(os.path.join(pwd, "checkpoints", "bert_dualG")):
    os.makedirs(os.path.join(pwd, "checkpoints", "bert_dualG"))

seed_file_name = os.path.join(pwd, "checkpoints", "bert_dualG")+f"seed_{timestamp}.txt"

# Save the initial seed to the file
with open(seed_file_name, "w") as seed_file:
    seed_file.write(f"Generated Seed: {seed}\n")
    seed_file.write("Configurations run:\n")  # Header for configurations


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
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

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


g_and_d_loss_checkpoint_config =[
    # With K.D losses - Either cosine or KL ; Without PreTrain loss included in D loss
    # {   "combination" : "G2_cos_D_baseline_3_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd",
    # "total_g_loss" : {"g_loss":0.0, "g_cosine_loss":1.00,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.10, "fake_loss":0.90, "fake_loss_pretrain":0.00} 
    # },
    #    {   "combination" : "G2_kl_D_baseline_3_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd",
    # "total_g_loss" : {"g_loss":0.0, "g_cosine_loss":0.00,"g_kl_loss":1.00}, 
    # "d_loss" : {"real_loss":0.10, "fake_loss":0.90, "fake_loss_pretrain":0.00} 
    # },
    #  {   "combination" : "G2_cos_kl_1_D_baseline_3_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd",
    # "total_g_loss" : {"g_loss":0.0, "g_cosine_loss":0.50,"g_kl_loss":0.50}, 
    # "d_loss" : {"real_loss":0.10, "fake_loss":0.90, "fake_loss_pretrain":0.00} 
    # },
    # {   "combination" : "G2_cos_kl_2_D_baseline_3_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd",
    # "total_g_loss" : {"g_loss":0.0, "g_cosine_loss":0.75,"g_kl_loss":0.25}, 
    # "d_loss" : {"real_loss":0.10, "fake_loss":0.90, "fake_loss_pretrain":0.00} 
    # },
    # {   "combination" : "G2_cos_kl_3_D_baseline_3_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd",
    # "total_g_loss" : {"g_loss":0.0, "g_cosine_loss":0.25,"g_kl_loss":0.75}, 
    # "d_loss" : {"real_loss":0.10, "fake_loss":0.90, "fake_loss_pretrain":0.00} 
    # },
    # # With MLE and K.D losses - Either cosine or KL ; ; Without PreTrain loss included in D loss
    # {   "combination" : "G1_G2_cos_1_D_baseline_3_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd",
    # "total_g_loss" : {"g_loss":0.50, "g_cosine_loss":0.50,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.10, "fake_loss":0.90, "fake_loss_pretrain":0.00} 
    # },
    # {   "combination" : "G1_G2_cos_2_D_baseline_3_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd",
    # "total_g_loss" : {"g_loss":0.75, "g_cosine_loss":0.25,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.10, "fake_loss":0.90, "fake_loss_pretrain":0.00} 
    # },
    # {   "combination" : "G1_G2_cos_3_D_baseline_3_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd",
    # "total_g_loss" : {"g_loss":0.25, "g_cosine_loss":0.75,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.10, "fake_loss":0.90, "fake_loss_pretrain":0.00} 
    # },
    # {   "combination" : "G1_G2_kl_1_D_baseline_3_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd",
    # "total_g_loss" : {"g_loss":0.50, "g_cosine_loss":0.00,"g_kl_loss":0.50}, 
    # "d_loss" : {"real_loss":0.10, "fake_loss":0.90, "fake_loss_pretrain":0.00} 
    # },
    # {   "combination" : "G1_G2_kl_2_D_baseline_3_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd",
    # "total_g_loss" : {"g_loss":0.75, "g_cosine_loss":0.00,"g_kl_loss":0.25}, 
    # "d_loss" : {"real_loss":0.10, "fake_loss":0.90, "fake_loss_pretrain":0.00} 
    # },
    # {   "combination" : "G1_G2_kl_3_D_baseline_3_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd",
    # "total_g_loss" : {"g_loss":0.75, "g_cosine_loss":0.00,"g_kl_loss":0.25}, 
    # "d_loss" : {"real_loss":0.10, "fake_loss":0.90, "fake_loss_pretrain":0.00} 
    # }, 
    # # With K.D losses - Either cosine or KL ; With PreTrain loss included in D loss
    # {   "combination" : "G2_cos_D_pretrain_1_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd",
    # "total_g_loss" : {"g_loss":0.0, "g_cosine_loss":1.00,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.25, "fake_loss":0.50, "fake_loss_pretrain":0.25} 
    # },
    #    {   "combination" : "G2_kl_D_pretrain_1_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd",
    # "total_g_loss" : {"g_loss":0.0, "g_cosine_loss":0.00,"g_kl_loss":1.00}, 
    # "d_loss" : {"real_loss":0.25, "fake_loss":0.50, "fake_loss_pretrain":0.25} 
    # },
    #  {   "combination" : "G2_cos_kl_1_D_pretrain_1_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd",
    # "total_g_loss" : {"g_loss":0.0, "g_cosine_loss":0.50,"g_kl_loss":0.50}, 
    # "d_loss" : {"real_loss":0.25, "fake_loss":0.50, "fake_loss_pretrain":0.25} 
    # },
    # {   "combination" : "G2_cos_kl_2_D_pretrain_1_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd",
    # "total_g_loss" : {"g_loss":0.0, "g_cosine_loss":0.75,"g_kl_loss":0.25}, 
    # "d_loss" : {"real_loss":0.25, "fake_loss":0.50, "fake_loss_pretrain":0.25} 
    # },
    # {   "combination" : "G2_cos_kl_3_D_pretrain_1_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd",
    # "total_g_loss" : {"g_loss":0.0, "g_cosine_loss":0.25,"g_kl_loss":0.75}, 
    # "d_loss" : {"real_loss":0.25, "fake_loss":0.50, "fake_loss_pretrain":0.25} 
    # },
    # # With MLE and K.D losses - Either cosine or KL ; ; With PreTrain loss included in D loss
    # {   "combination" : "G1_G2_cos_1_D_baseline_3_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd",
    # "total_g_loss" : {"g_loss":0.50, "g_cosine_loss":0.50,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.25, "fake_loss":0.50, "fake_loss_pretrain":0.25} 
    # },
    # {   "combination" : "G1_G2_cos_2_D_pretrain_1_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd",
    # "total_g_loss" : {"g_loss":0.75, "g_cosine_loss":0.25,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.25, "fake_loss":0.50, "fake_loss_pretrain":0.25} 
    # },
    # {   "combination" : "G1_G2_cos_3_D_pretrain_1_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd",
    # "total_g_loss" : {"g_loss":0.25, "g_cosine_loss":0.75,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.25, "fake_loss":0.50, "fake_loss_pretrain":0.25} 
    # },
    # {   "combination" : "G1_G2_kl_1_D_pretrain_1_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd",
    # "total_g_loss" : {"g_loss":0.50, "g_cosine_loss":0.00,"g_kl_loss":0.50}, 
    # "d_loss" : {"real_loss":0.25, "fake_loss":0.50, "fake_loss_pretrain":0.25} 
    # },
    # {   "combination" : "G1_G2_kl_2_D_pretrain_1_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd",
    # "total_g_loss" : {"g_loss":0.75, "g_cosine_loss":0.00,"g_kl_loss":0.25}, 
    # "d_loss" : {"real_loss":0.25, "fake_loss":0.50, "fake_loss_pretrain":0.25} 
    # },
    # {   "combination" : "G1_G2_kl_3_D_pretrain_1_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd",
    # "total_g_loss" : {"g_loss":0.75, "g_cosine_loss":0.00,"g_kl_loss":0.25}, 
    # "d_loss" : {"real_loss":0.25, "fake_loss":0.50, "fake_loss_pretrain":0.25} 
    # }
    #### Without K.D losses - Without PreTrain loss included in D loss
    # {   "combination" : "G1_D_NoPreTrain_baseline_1_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd",
    # "total_g_loss" : {"g_loss":1.00, "g_cosine_loss":0.00,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.75, "fake_loss":0.25, "fake_loss_pretrain":0.00} 
    # },
    # Combinations of above with different loss weights for fake_loss_pretrain
    # {   "combination" : "G1_75_25_D_PreTrain_1_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd",
    # "total_g_loss" : {"g_loss":1.00, "g_cosine_loss":0.00,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.75, "fake_loss":0.25, "fake_loss_pretrain":0.25} 
    # },
    # {   "combination" : "G1_75_25_D_PreTrain_2_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd",
    # "total_g_loss" : {"g_loss":1.00, "g_cosine_loss":0.00,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.75, "fake_loss":0.25, "fake_loss_pretrain":0.50} 
    # },
    # {   "combination" : "G1_75_25_D_PreTrain_3_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd",
    # "total_g_loss" : {"g_loss":1.00, "g_cosine_loss":0.00,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.75, "fake_loss":0.25, "fake_loss_pretrain":0.75} 
    # }, ### Dec-2024
    #    {   "combination" : "G_0_25_75_cos_kl_75_25_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":0.25,"g_kl_loss":0.75}, 
    # "d_loss" : {"real_loss":0.75, "fake_loss":0.25, "fake_loss_pretrain":0.00} 
    # },
    # {   "combination" : "G_0_25_75_cos_kl_50_50_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":0.25,"g_kl_loss":0.75}, 
    # "d_loss" : {"real_loss":0.50, "fake_loss":0.50, "fake_loss_pretrain":0.00} 
    # },
    # {   "combination" : "G_0_25_75_cos_kl_25_75_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":0.25,"g_kl_loss":0.75}, 
    # "d_loss" : {"real_loss":0.25, "fake_loss":0.75, "fake_loss_pretrain":0.00} 
    # },
    # {   "combination" : "G_0_25_75_cos_kl_10_90_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":0.25,"g_kl_loss":0.75}, 
    # "d_loss" : {"real_loss":0.10, "fake_loss":0.90, "fake_loss_pretrain":0.00} 
    # },
    # {   "combination" : "G_0_25_75_cos_kl_25_75_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":0.25,"g_kl_loss":0.75}, 
    # "d_loss" : {"real_loss":0.90, "fake_loss":0.10, "fake_loss_pretrain":0.00} 
    # },
    # {   "combination" : "G_0_25_75_cos_kl_1_0_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":0.25,"g_kl_loss":0.75}, 
    # "d_loss" : {"real_loss":1.00, "fake_loss":0.00, "fake_loss_pretrain":0.00} 
    # },
    # {   "combination" : "G_0_25_75_cos_kl_0_1_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":0.25,"g_kl_loss":0.75}, 
    # "d_loss" : {"real_loss":0.00, "fake_loss":1.00, "fake_loss_pretrain":0.00} 
    # },
    # {   "combination" : "G_0_25_75_cos_kl_0_0_1_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":0.25,"g_kl_loss":0.75}, 
    # "d_loss" : {"real_loss":0.00, "fake_loss":0.00, "fake_loss_pretrain":1.00} 
    # },
    # {   "combination" : "G_0_25_75_cos_kl_20_60_20_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":0.25,"g_kl_loss":0.75}, 
    # "d_loss" : {"real_loss":0.20, "fake_loss":0.60, "fake_loss_pretrain":0.20} 
    # },
    # {   "combination" : "G_0_25_75_cos_kl_40_40_20_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":0.25,"g_kl_loss":0.75}, 
    # "d_loss" : {"real_loss":0.40, "fake_loss":0.40, "fake_loss_pretrain":0.20} 
    # },
    # {   "combination" : "G_0_25_75_cos_kl_40_20_40_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":0.25,"g_kl_loss":0.75}, 
    # "d_loss" : {"real_loss":0.40, "fake_loss":0.20, "fake_loss_pretrain":0.40} 
    # },
    # {   "combination" : "G_0_25_75_cos_kl_80_10_10_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":0.25,"g_kl_loss":0.75}, 
    # "d_loss" : {"real_loss":0.80, "fake_loss":0.10, "fake_loss_pretrain":0.10} 
    # },
    # {   "combination" : "G_0_25_75_cos_kl_10_80_10_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":0.25,"g_kl_loss":0.75}, 
    # "d_loss" : {"real_loss":0.10, "fake_loss":0.80, "fake_loss_pretrain":0.10} 
    # },
    # {   "combination" : "G_0_25_75_cos_kl_10_10_80_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":0.25,"g_kl_loss":0.75}, 
    # "d_loss" : {"real_loss":0.10, "fake_loss":0.10, "fake_loss_pretrain":0.80} 
    # }
    #   {   "combination" : "G_0_25_75_cos_kl_25_50_25_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_80",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":0.25,"g_kl_loss":0.75}, 
    # "d_loss" : {"real_loss":0.25, "fake_loss":0.50, "fake_loss_pretrain":0.25} 
    # },
    # {   "combination" : "G_0_50_50_cos_kl_25_50_25_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_80",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":0.50,"g_kl_loss":0.50}, 
    # "d_loss" : {"real_loss":0.25, "fake_loss":0.50, "fake_loss_pretrain":0.25} 
    # },
    # {   "combination" : "G_0_50_50_cos_kl_10_90_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_80",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":0.50,"g_kl_loss":0.50}, 
    # "d_loss" : {"real_loss":0.10, "fake_loss":0.90, "fake_loss_pretrain":0.00} 
    # },
    # { "combination" : "G_0_50_50_cos_kl_10_90_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100_10PG_0001lr",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":0.50,"g_kl_loss":0.50}, 
    # "d_loss" : {"real_loss":0.10, "fake_loss":0.90, "fake_loss_pretrain":0.00} 
    # }
    # { "combination" : "G_50_50_0_cos_kl_10_90_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100_10PG_0001lr",
    # "total_g_loss" : {"g_loss":0.50, "g_cosine_loss":0.50,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.10, "fake_loss":0.90, "fake_loss_pretrain":0.00} 
    # }
    # { "combination" : "G_25_75_0_cos_kl_10_90_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100_10PG_0001lr",
    # "total_g_loss" : {"g_loss":0.25, "g_cosine_loss":0.75,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.10, "fake_loss":0.90, "fake_loss_pretrain":0.00} 
    # }
    # { "combination" : "G_25_75_0_cos_kl_10_90_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100_10PG_0001lr",
    # "total_g_loss" : {"g_loss":0.25, "g_cosine_loss":0.75,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.10, "fake_loss":0.90, "fake_loss_pretrain":0.00} 
    # }
    # { "combination" : "G_10_90_0_cos_kl_10_90_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100_10PG_0001lr",
    # "total_g_loss" : {"g_loss":0.10, "g_cosine_loss":0.90,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.10, "fake_loss":0.90, "fake_loss_pretrain":0.00} 
    # }
    # { "combination" : "G_10_90_0_cos_kl_10_90_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100_10PG_0001lr_db",
    # "total_g_loss" : {"g_loss":0.10, "g_cosine_loss":0.90,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.10, "fake_loss":0.90, "fake_loss_pretrain":0.00} 
    # }
    # {   "combination" : "G_0_75_25_cos_kl_10_90_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":0.75,"g_kl_loss":0.25}, 
    # "d_loss" : {"real_loss":0.10, "fake_loss":0.90, "fake_loss_pretrain":0.00} 
    # },
    #     {   "combination" : "G_0_75_25_cos_kl_25_50_25_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":0.75,"g_kl_loss":0.25}, 
    # "d_loss" : {"real_loss":0.25, "fake_loss":0.50, "fake_loss_pretrain":0.25} 
    # }
    # {   "combination" : "G_1_0_0_cos_kl_0_0_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100",
    # "total_g_loss" : {"g_loss":1.00, "g_cosine_loss":0.00,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.00, "fake_loss":0.00, "fake_loss_pretrain":0.00} 
    # }
    # {"combination" : "G_1_0_0_cos_kl_0_1_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100_10PG_0001lr",
    # "total_g_loss" : {"g_loss":1.00, "g_cosine_loss":0.00,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.00, "fake_loss":1.00, "fake_loss_pretrain":0.00} 
    # }
    # {   "combination" : "G_1_0_0_cos_kl_1_0_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100",
    # "total_g_loss" : {"g_loss":1.00, "g_cosine_loss":0.00,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":1.00, "fake_loss":0.00, "fake_loss_pretrain":0.00} 
    # },
    # {   "combination" : "G_1_0_0_cos_kl_0_0_1_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100",
    # "total_g_loss" : {"g_loss":1.00, "g_cosine_loss":0.00,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.00, "fake_loss":0.00, "fake_loss_pretrain":1.00} 
    # }
    # {   "combination" : "G_1_0_0_cos_kl_25_50_25_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100",
    # "total_g_loss" : {"g_loss":1.00, "g_cosine_loss":0.00,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.00, "fake_loss":0.00, "fake_loss_pretrain":1.00} 
    # }
    ############################
    # {   "combination" : "G1_D_NoPreTrain_baseline_2_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd",
    # "total_g_loss" : {"g_loss":1.00, "g_cosine_loss":0.00,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.25, "fake_loss":0.75, "fake_loss_pretrain":0.00} 
    # },
    # {   "combination" : "G1_D_NoPreTrain_baseline_3_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd",
    # "total_g_loss" : {"g_loss":1.00, "g_cosine_loss":0.00,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.50, "fake_loss":0.50, "fake_loss_pretrain":0.00} 
    # }
    # { "combination" : "G_10_90_0_cos_kl_10_90_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100_10PG_00001lr",
    # "total_g_loss" : {"g_loss":0.10, "g_cosine_loss":0.90,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.10, "fake_loss":0.90, "fake_loss_pretrain":0.00} 
    # }
    # { "combination" : "G_25_75_0_cos_kl_10_90_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100_100PG_00001lr",
    # "total_g_loss" : {"g_loss":0.25, "g_cosine_loss":0.75,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.10, "fake_loss":0.90, "fake_loss_pretrain":0.00} 
    # }
    # { "combination" : "G_0_1_0_cos_kl_0_0_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100_0PG_0001lr",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":1.00,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.00, "fake_loss":0.00, "fake_loss_pretrain":0.00} 
    # },
    # { "combination" : "G_0_0_1_cos_kl_0_0_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100_0PG_0001lr",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":0.00,"g_kl_loss":1.00}, 
    # "d_loss" : {"real_loss":0.00, "fake_loss":0.00, "fake_loss_pretrain":0.00} 
    # },
    # { "combination" : "G_0_50_50_cos_kl_0_0_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100_0PG_0001lr",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":0.50,"g_kl_loss":0.50}, 
    # "d_loss" : {"real_loss":0.00, "fake_loss":0.00, "fake_loss_pretrain":0.00} 
    # }
    # { "combination" : "G_0_1_0_cos_kl_10_100_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100_100PG_00001lr",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":1.00,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":1, "fake_loss":10, "fake_loss_pretrain":0.00} 
    # }
    # { "combination" : "G_0_0_1_cos_kl_10_90_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100_10PG_0001lr",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":0.00,"g_kl_loss":1.00}, 
    # "d_loss" : {"real_loss":0.10, "fake_loss":0.90, "fake_loss_pretrain":0.00} 
    # }
    # { "combination" : "G_0_50_50_cos_kl_0_0_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100_0PG_0001lr",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":0.50,"g_kl_loss":0.50}, 
    # "d_loss" : {"real_loss":0.00, "fake_loss":0.00, "fake_loss_pretrain":0.00} 
    # }
    # { "combination" : "G_50_100_0_cos_kl_10_90_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100_100PG_00001lr",
    # "total_g_loss" : {"g_loss":0.50, "g_cosine_loss":10.00,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.1, "fake_loss":0.9, "fake_loss_pretrain":0.00} 
    # }
    # { "combination" : "G_10_500_0_cos_kl_10_90_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100_100PG_00001lr",
    # "total_g_loss" : {"g_loss":0.10, "g_cosine_loss":50.00,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.1, "fake_loss":0.9, "fake_loss_pretrain":0.00} 
    # }
    # { "combination" : "G_50_100_0_cos_kl_10_90_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100_1000PG_00001lr",
    # "total_g_loss" : {"g_loss":0.50, "g_cosine_loss":10.00,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.1, "fake_loss":0.9, "fake_loss_pretrain":0.00} 
    # }
    #     { "combination" : "G_50_100_0_cos_kl_10_70_20_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100_1000PG_00001lr",
    # "total_g_loss" : {"g_loss":0.50, "g_cosine_loss":10.00,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.1, "fake_loss":0.7, "fake_loss_pretrain":0.20} 
    # },
    #         { "combination" : "G_50_100_0_cos_kl_10_100_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100_1000PG_00001lr",
    # "total_g_loss" : {"g_loss":0.50, "g_cosine_loss":10.00,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.1, "fake_loss":10.00, "fake_loss_pretrain":0.00} 
    # }
    # { "combination" : "G_50_100_0_cos_kl_10_100_100_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100_1000PG_00001lr",
    # "total_g_loss" : {"g_loss":0.50, "g_cosine_loss":10.00,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.1, "fake_loss":10, "fake_loss_pretrain":10} 
    # }
    # { "combination" : "G_0.5_10_0_cos_kl_0.1_1_1_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100_1000PG_00001lr",
    # "total_g_loss" : {"g_loss":0.50, "g_cosine_loss":10.00,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.1, "fake_loss":1, "fake_loss_pretrain":1} 
    # }
    #    { "combination" : "G_0.5_10_0_cos_kl_0.1_1_1_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100_10000PG_00001lr",
    # "total_g_loss" : {"g_loss":0.50, "g_cosine_loss":10.00,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.1, "fake_loss":0.7, "fake_loss_pretrain":0.2} 
    # }
    # { "combination" : "G_0.5_10_0_cos_kl_0.1_0.7_0.2_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100_5000PG_00001lr",
    # "total_g_loss" : {"g_loss":0.50, "g_cosine_loss":10.00,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.1, "fake_loss":0.7, "fake_loss_pretrain":0.2} 
    # }
    #     { "combination" : "G_0.5_10_0_cos_kl_0.1_0.7_0.2_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100_2000PG_00001lr",
    # "total_g_loss" : {"g_loss":0.50, "g_cosine_loss":10.00,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.1, "fake_loss":0.7, "fake_loss_pretrain":0.2} 
    # }
    #       { "combination" : "G_0.5_10_0_cos_kl_0.1_0.7_0.2_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100_1000PG_00001lr",
    # "total_g_loss" : {"g_loss":0.50, "g_cosine_loss":10.00,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.1, "fake_loss":0.7, "fake_loss_pretrain":0.5} 
    # }
    # Including P.G loss and revers KLD with logits 
    # { "combination" : "G_0_0_0_0_1_cos_kl_pg_rkldlgts_0_0_0_D_1000_to_1_mil_only_biasTermsUpd_PGloss_1_2_upd_bs_40_0PG_1rkld_lgts_00001lr_ep_5",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":00.00,"g_kl_loss":0.00, "g_pg_loss":0, "g_rkld_logits":10}, 
    # "d_loss" : {"real_loss":0.0, "fake_loss":0.0, "fake_loss_pretrain":0.0} 
    # }
    # { "combination" : "G_0_0_0_0_1_cos_kl_pg_rkldlgts_0_0_0_D_1000_to_1_mil_only_biasTermsUpd_PGloss_1_2_upd_bs_40_0PG_100rkld_lgts_00001lr_ep_5",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":00.00,"g_kl_loss":0.00, "g_pg_loss":0, "g_rkld_logits":100}, 
    # "d_loss" : {"real_loss":0.0, "fake_loss":0.0, "fake_loss_pretrain":0.0} 
    # }
    # { "combination" : "G_0_0_0_0_1_cos_kl_pg_rkldlgts_0_0_0_D_2000_to_1_mil_onlylm_lyr_frz_PGloss_1_2_upd_bs_40_0PG_100rkld_lgts_00001lr_ep_5",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":00.00,"g_kl_loss":0.00, "g_pg_loss":0, "g_rkld_logits":100}, 
    # "d_loss" : {"real_loss":0.0, "fake_loss":0.0, "fake_loss_pretrain":0.0} 
    # }
    # { "combination" : "G_0_0_0_0_1_cos_kl_pg_rkldlgts_0_0_0_D_x_to_1_mil_Bias_LM_PGloss_1_2_upd_bs_40_0PG_100rkld_lgts_00001lr_ep_5",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":00.00,"g_kl_loss":0.00, "g_pg_loss":0, "g_rkld_logits":100}, 
    # "d_loss" : {"real_loss":0.0, "fake_loss":0.0, "fake_loss_pretrain":0.0},
    # "gradient_update": {"BIAS": True, "LM": False},
    # "Dataset":{"train_size":1000},
    # "Mscll":{"Comments": "Baseline : This is the combination with Only Bias layers updating and LM layer freezed"}
    # },
    #  { "combination" : "G_0_0_0_0_1_cos_kl_pg_rkldlgts_0_0_0_D_x_to_1_mil_Bias_T_LM_T_PGloss_1_2_upd_bs_40_0PG_100rkld_lgts_00001lr_ep_5",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":00.00,"g_kl_loss":0.00, "g_pg_loss":0, "g_rkld_logits":100}, 
    # "d_loss" : {"real_loss":0.0, "fake_loss":0.0, "fake_loss_pretrain":0.0},
    # "gradient_update": {"BIAS": True, "LM": True},
    # "Dataset":{"train_size":1000},
    # "Mscll":{"Comments": "A : This is the combination with Only Bias layers updating and LM layer updating"}
    # },
    # { "combination" : "G_0_0_0_0_1_cos_kl_pg_rkldlgts_0_0_0_D_x_to_1_mil_Bias_T_LM_T_PGloss_1_2_upd_bs_40_0PG_100rkld_lgts_00001lr_ep_5",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":00.00,"g_kl_loss":0.00, "g_pg_loss":0, "g_rkld_logits":100}, 
    # "d_loss" : {"real_loss":0.0, "fake_loss":0.0, "fake_loss_pretrain":0.0},
    # "gradient_update": {"BIAS": True, "LM": True},
    # "Dataset":{"train_size":10000},
    # "Mscll":{"Comments": "A : This is the combination with Only Bias layers updating and LM layer updating"}
    # },
    #   { "combination" : "G_0_0_0_0_1_cos_kl_pg_rkldlgts_0_0_0_D_x_to_1_mil_Bias_F_LM_F_PGloss_1_2_upd_bs_40_0PG_100rkld_lgts_00001lr_ep_5",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":00.00,"g_kl_loss":0.00, "g_pg_loss":0, "g_rkld_logits":100}, 
    # "d_loss" : {"real_loss":0.0, "fake_loss":0.0, "fake_loss_pretrain":0.0},
    # "gradient_update": {"BIAS": False, "LM": False}, 
    # "Dataset":{"train_size":1000},
    # "Mscll":{"Comments": "B : This is the combination with ALL layers updating and LM layer freezed"}
    # },
    # { "combination" : "G_0_0_0_0_1_cos_kl_pg_rkldlgts_0_0_0_D_x_to_1_mil_Bias_F_LM_T_PGloss_1_2_upd_bs_40_0PG_100rkld_lgts_00001lr_ep_5",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":00.00,"g_kl_loss":0.00, "g_pg_loss":0, "g_rkld_logits":100}, 
    # "d_loss" : {"real_loss":0.0, "fake_loss":0.0, "fake_loss_pretrain":0.0},
    # "gradient_update": {"BIAS": False, "LM": True}, 
    # "Dataset":{"train_size":1000},
    # "Mscll":{"Comments": "C : This is the combination with ALL layers updating and LM layer updating"}
    # },
    # { "combination" : "G_0_0_0_0_1_cos_kl_pg_rkldlgts_0_0_0_D_x_to_1_mil_Bias_F_LM_T_PGloss_1_2_upd_bs_40_0PG_100rkld_lgts_00001lr_ep_5",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":00.00,"g_kl_loss":0.00, "g_pg_loss":0, "g_rkld_logits":100}, 
    # "d_loss" : {"real_loss":0.0, "fake_loss":0.0, "fake_loss_pretrain":0.0},
    # "gradient_update": {"BIAS": False, "LM": True}, 
    # "Dataset":{"train_size":2000},
    # "Mscll":{"Comments": "C : This is the combination with ALL layers updating and LM layer updating"}
    # },
    #   { "combination" : "G_0_0_0_0_1_cos_kl_pg_rkldlgts_0_0_0_D_x_to_1_mil_Bias_T_LM_T_PGloss_1_2_upd_bs_40_0PG_100rkld_lgts_00001lr_ep_5",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":00.00,"g_kl_loss":0.00, "g_pg_loss":0, "g_rkld_logits":100}, 
    # "d_loss" : {"real_loss":0.0, "fake_loss":0.0, "fake_loss_pretrain":0.0},
    # "gradient_update": {"BIAS": True, "LM": True},
    # "Dataset":{"train_size":1000},
    # "Mscll":{"Comments": "A : This is the combination with Only Bias layers updating and LM layer updating"}
    # },
    # { "combination" : "G_0_0_0_0_1_cos_kl_pg_rkldlgts_0_0_0_D_x_to_1_mil_Bias_T_LM_T_PGloss_1_2_upd_bs_40_0PG_100rkld_lgts_00001lr_ep_5",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":00.00,"g_kl_loss":0.00, "g_pg_loss":0, "g_rkld_logits":100}, 
    # "d_loss" : {"real_loss":0.0, "fake_loss":0.0, "fake_loss_pretrain":0.0},
    # "gradient_update": {"BIAS": True, "LM": True},
    # "Dataset":{"train_size":2000},
    # "Mscll":{"Comments": "A : This is the combination with Only Bias layers updating and LM layer updating"}
    # },
    # { "combination" : "G_0_0_0_0_1_cos_kl_pg_rkldlgts_0_0_0_D_x_to_1_mil_Bias_LM_PGloss_1_2_upd_bs_40_0PG_100rkld_lgts_00001lr_ep_5",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":00.00,"g_kl_loss":0.00, "g_pg_loss":0, "g_rkld_logits":100}, 
    # "d_loss" : {"real_loss":0.0, "fake_loss":0.0, "fake_loss_pretrain":0.0},
    # "gradient_update": {"BIAS": True, "LM": False},
    # "Dataset":{"train_size":2000},
    # "Mscll":{"Comments": "Baseline : This is the combination with Only Bias layers updating and LM layer freezed"}
    # }
    #     { "combination" : "G_0_0_0_0_1_cos_kl_pg_rkldlgts_0_0_0_D_x_to_1_mil_Bias_LM_PGloss_1_2_upd_bs_40_0PG_100rkld_lgts_00001lr_ep_5",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":00.00,"g_kl_loss":0.00, "g_pg_loss":0, "g_rkld_logits":100}, 
    # "d_loss" : {"real_loss":0.0, "fake_loss":0.0, "fake_loss_pretrain":0.0},
    # "gradient_update": {"BIAS": True, "LM": False},
    # "Dataset":{"train_size":2000},
    # "Mscll":{"Comments": "Baseline : This is the combination with Only Bias layers updating and LM layer freezed"}
    # }
    { "combination" : "G_0_0_0_0_1_cos_kl_pg_rkldlgts_0_0_0_D_x_to_1_mil_Bias_T_LM_T_PGloss_1_2_upd_bs_40_0PG_100rkld_lgts_00001lr_ep_5",
    "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":00.00,"g_kl_loss":0.00, "g_pg_loss":0, "g_rkld_logits":100}, 
    "d_loss" : {"real_loss":0.0, "fake_loss":0.0, "fake_loss_pretrain":0.0},
    "gradient_update": {"BIAS": True, "LM": False},
    "Dataset":{"train_size":1000},
    "Mscll":{"Comments": "Baseline : This is the combination with Only Bias layers updating and LM layer freezed"}
    }
    # { "combination" : "G_0_0_0_0_1_cos_kl_pg_rkldlgts_0_0_0_D_x_to_1_mil_Bias_T_LM_T_PGloss_1_2_upd_bs_40_0PG_100rkld_lgts_00001lr_ep_5_peft",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":00.00,"g_kl_loss":0.00, "g_pg_loss":0, "g_rkld_logits":100}, 
    # "d_loss" : {"real_loss":0.0, "fake_loss":0.0, "fake_loss_pretrain":0.0},
    # "gradient_update": {"BIAS": False, "LM": False, "LORA":True},
    # "Dataset":{"train_size":1000},
    # "Mscll":{"Comments": "LORA Baseline  : This is the combination with Bias layers not updating and LM layer freezed, with LORA updating"}
    # }
]

def main(args, config):

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
    # train_dataset = dataset["train"].select(range(1000000))
    # train_dataset = dataset["train"].select(range(100000))
    # train_dataset = dataset["train"].select(range(1000))
    train_dataset = dataset["train"].select(range(config['Dataset']['train_size']))

    # Keep the full valid and test datasets
    valid_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    # Pre-processing the data
    # To-Do : Need to change the max_length to 50 from 128
    source_lang = "en"
    target_lang = "fr"
    prefix = ""

    from transformers import AutoTokenizer

    checkpoint = 'Helsinki-NLP/opus-mt-en-fr'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def preprocess_MarianMT(examples):

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

    """
    ### -- Adding LORA and Adapter layers -------------- ####
    
    # LoRA configuration
    from peft import get_peft_model, LoraConfig, TaskType
    from peft import LoraConfig

    lora_config = LoraConfig(
        task_type="SEQ_2_SEQ_LM",  # This task type aligns with MarianMTModel
        r=8,  # LoRA rank
        lora_alpha=16,  # Scaling factor
        lora_dropout=0.1,  # Dropout probability
        target_modules=[
            "self_attn.k_proj", "self_attn.q_proj", "self_attn.v_proj",  # Encoder self-attention
            "encoder_attn.k_proj", "encoder_attn.q_proj", "encoder_attn.v_proj"  # Decoder encoder-attention
        ]
    )
    """
    
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
        generator1_pretrained.eval()

        # Initialize loss for this epoch # Usually, the losses for G and D are not calculated during the training phase, but just the model parameters are updated. However, we are just capturing these metrics for analysis purposes.
        total_train_g_loss = 0
        total_train_d_loss = 0

        # Freeze embedding layers 
        #Access the underlying MarianMT model from DataParallel wrapper if necessary
        generator2_train = generator2_train.module if hasattr(generator2_train, 'module') else generator2_train
        """
        generator2_train = get_peft_model(generator2_train, lora_config)
        """
        
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

        # # Freeze early encoder layers (for example, the first 2 layers)
        # for layer in generator2_train.model.encoder.layers[:2]:
        #     for param in layer.parameters():
        #         param.requires_grad = False
        
        # Similarly, you might consider freezing early decoder layers if needed
        # for layer in generator2_train.model.decoder.layers[:2]:
        #     for param in layer.parameters():
        #         param.requires_grad = False
                
        """

        # Freeze the generator2_train model lm layers based on counter
        # lm_counter = 0
        # lm_interval = 10000  # Interval for updating the lm layers


        # print("generator2_train ", generator2_train)
        # writer = SummaryWriter()
        # # generator2_train.to_text_file("generator2_train.txt")
        # for name, param in generator2_train.model.parameters():
        
        
        ############################# BIAS LAYERS ########################################
        for name, param in generator2_train.named_parameters():
            param.requires_grad = False
            
            # if "lora_" in name:
            #     param.requires_grad = config['gradient_update']['LORA']
            
            # Enable requires_grad only for bias terms
            if 'bias' in name:
                param.requires_grad = config['gradient_update']['BIAS'] #True

        # # Freeze early encoder layers (for example, the first 2 layers)
        # for layer in generator2_train.model.encoder.layers[:1]:
        #     for param in layer.parameters():
        #         param.requires_grad = True
        
        
        # for layer in generator2_train.model.decoder.layers[:2]:
        #     for param in layer.parameters():
        #         param.requires_grad = True


        ########################### LM LAYERS ########################################
        for param in generator2_train.lm_head.parameters():
            param.requires_grad = config['gradient_update']['LM'] #False
        
        # Update logic
        update_count = 0
        pg_count = 0  # Counter for policy gradient inclusion
        pg_interval = 2  # Interval for PG loss calculation (every 2 updates)

        """
        if torch.cuda.device_count() > 1:
            generator2_train = torch.nn.DataParallel(generator2_train).cuda()
        """

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

             # Access the original TransformerModel from the DataParallel wrapper
            generator2_train_dp = generator2_train.module if isinstance(generator2_train, torch.nn.DataParallel) else generator2_train

            generator2_train_out = generator2_train_dp(input_ids=src_sentences, attention_mask=attention_mask, decoder_input_ids=tgt_sentences , output_hidden_states=True, return_dict=True)
            # print("generator2_train_out shape ", generator2_train_out.shape)
            # print("type of generator2_train_out", type(generator2_train_out))
            print(" generator2_train_keys() ", generator2_train_out.keys())
            # print("generator2_train_out to_tuple", generator2_train_out.to_tuple())
            # dict_keys(['logits', 'past_key_values', 'decoder_hidden_states', 'encoder_last_hidden_state', 'encoder_hidden_states'])
            print("generator2_train_out logits shape ", generator2_train_out.logits.shape)
            # print("generator2_train_out decoder_hidden_states shape ", generator2_train_out.decoder_hidden_states)
            print("generator2_train_out encoder_last_hidden_state shape ", generator2_train_out.encoder_last_hidden_state.shape)

            fake_tgt_sentences_probs = F.log_softmax(generator2_train_out.logits, dim=-1)

            fake_tgt_sentences_probs = fake_tgt_sentences_probs.view(
                -1, fake_tgt_sentences_probs.size(-1)
            )  # Shape: [batch_size * seq_len, vocab_size]
            tgt_sentences_flat = tgt_sentences.view(-1)  # Shape: [batch_size * seq_len]
           
            print("shape of generator2_train_out.logits ", generator2_train_out.logits.shape)

            logits_flat = generator2_train_out.logits.view(-1, generator2_train_out.logits.size(-1))  # Shape: [batch_size * seq_len, vocab_size]
            
            print("shape of tgt_sentences ", tgt_sentences.shape)
            print(" shape of logits_flat ", logits_flat.shape)

            g_loss = g_criterion(logits_flat, tgt_sentences_flat)


            # -------------------------------------Generating translations using the pre-trained generator
            src_sentences_for_G1 = ids_to_sentences_bert(src_sentences)
            print("At epoch_i ", epoch_i)
            print("At batch no# ", i)
            print("generating translations using the pre-trained generator")

             # Access the original TransformerModel from the DataParallel wrapper
            original_generator_pt = generator1_pretrained.module if isinstance(generator1_pretrained, torch.nn.DataParallel) else generator1_pretrained
            
            # OLD METHOD TO TRANSLATE SENTENCES FOR DISCRIMINATOR
            translated_sentences_from_G1 = original_generator_pt.translate(
                src_sentences_for_G1
            )
            
            # computing reverse KLD with logits 
            generator_teacher_logits_list = extract_fairseq_logits(original_generator_pt, train_dataset, src_sentences_for_G1)
            
            fs_vocab_size = len(original_generator_pt.task.source_dictionary) #44512   # Fairseq vocabulary size
            marian_vocab_size = generator2_train_dp.config.vocab_size #59514  # Marian vocabulary size
            
            reverse_kld_values = reverse_kld_logits(generator_teacher_logits_list, generator2_train_out.logits, fs_vocab_size, marian_vocab_size)
            
            print("shape of reverse kld values ", reverse_kld_values.shape)
            # Ensure reverse_kld_values is a scalar or aggregated tensor if required
            reverse_kld_mean = reverse_kld_values.mean()  # Example: take the mean if needed
            
            # Convert the translated sentences back into token IDs and attention masks
            fake_tgt_sentences_G1_pretrain_probs = sentences_to_ids(
                translated_sentences_from_G1, max_length=128
            )

            # -------------------------------------------------Train the discriminator
            print("training discriminator")
            print("At epoch_i ", epoch_i)
            print("At batch no# ", i)
            optimizer_d.zero_grad()
            
            # Ensure targets for real and fake are correctly shaped
            real_targets = torch.ones(src_sentences.size(0), 1).to(device)  # Real
            fake_targets = torch.zeros(src_sentences.size(0), 1).to(device)  # Fake
            
            # --------------------------------Real loss of the discriminator --------------
            real_loss = d_criterion(
                discriminator_cnn(src_sentences, tgt_sentences),
                real_targets,
            )

            # -----------------------------------------------------Generator 2 Train()-----------------------------------------
            print("At epoch_i ", epoch_i)
            print("At batch no# ", i)
            print("training generator 2")
        
            _, prediction = fake_tgt_sentences_probs.topk(1)
            prediction = prediction.squeeze(1)
            fake_tgt_sentences = torch.reshape(prediction, src_sentences.shape)
            
            # ----------------------------------------- Generator 1 Pre-Trained ---------------------------------------------
            g2_token_embeds = generator2_train_dp.model.shared(fake_tgt_sentences)
            g1_token_embeds = generator2_train_dp.model.shared(fake_tgt_sentences_G1_pretrain_probs['input_ids'].to(device))

            print("shape of g2 token embeds: ", g2_token_embeds.shape)
            print("shape of g1 token embeds: ", g1_token_embeds.shape)

            g_cosine_loss = cosine_embedding_loss(g2_token_embeds, g1_token_embeds)
            g_kl_loss_reverse = kl_divergence_loss_reverse(g2_token_embeds, g1_token_embeds)
            g_kl_loss = kl_divergence_loss(g2_token_embeds, g1_token_embeds)


            # #################################################   Total G2 loss
            # Check if it's time to calculate PG loss based on pg_count
            if pg_count % pg_interval == 0:
                rewards = discriminator_cnn(src_sentences, fake_tgt_sentences.detach())
                print("type of rewards ", type(rewards))
                print("rewards shape ", rewards.shape)
                pg_loss = policy_gradient_loss(discriminator_cnn, src_sentences, fake_tgt_sentences, rewards)
                total_g_loss = config['total_g_loss']['g_loss']*g_loss + config['total_g_loss']['g_cosine_loss']*g_cosine_loss + config['total_g_loss']['g_kl_loss']*g_kl_loss + config['total_g_loss']['g_pg_loss']* pg_loss + config['total_g_loss']['g_rkld_logits']*reverse_kld_mean # "g_pg_loss":1000, "g_rkld_logits":1
            else:
                total_g_loss = config['total_g_loss']['g_loss']*g_loss + config['total_g_loss']['g_cosine_loss']*g_cosine_loss + config['total_g_loss']['g_kl_loss']*g_kl_loss + config['total_g_loss']['g_rkld_logits']*reverse_kld_mean

            pg_count += 1  # Increment PG counter

            if update_count % 1 == 0:
                total_g_loss.backward()
                optimizer_g.step()
                optimizer_g.zero_grad()  # Clear gradients after update
                total_train_g_loss += total_g_loss.item() 
            else:
                with torch.no_grad():
                    total_g_loss.backward()  # Accumulate gradients without stepping
            update_count += 1  # Increment the update count

            # ---------------------------------------- fake loss from Generator 2 Train()--------------------------
            fake_tgt_sentences = fake_tgt_sentences.to(device)
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
                        
            import sqlite3
            
            def init_db_train():
                conn = None
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
                    if conn:
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
                conn = None
                try:
                    conn = sqlite3.connect(db_name)
                    c = conn.cursor()
                    size_of_src_sentences_converted_logging_org = len(src_sentences_converted_logging_org)
                    epoch_i_list = [epoch_i] * size_of_src_sentences_converted_logging_org
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
                    if conn:
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

            d_loss = config['d_loss']['real_loss']*real_loss + config['d_loss']['fake_loss']*fake_loss + config['d_loss']['fake_loss_pretrain']*fake_loss_pretrain

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

        torch.save(
                generator2_train,
                open(checkpoints_path + f"train_checkpoint_generator_dill_open_format_at_{epoch_i}.pt", "wb"),
                pickle_module=dill,
            )
        torch.save(
                discriminator_cnn,
                open(checkpoints_path + f"train_checkpoint_discriminator_dill_direct_format_at_{epoch_i}.pt", "wb"),
                pickle_module=dill,
            )
        generator2_train.save_pretrained(checkpoints_path + f"train_checkpoint_generator_save_pretrained_at_{epoch_i}")
        tokenizer.save_pretrained(checkpoints_path + f"train_checkpoint_tokenizer_save_pretrained_at_{epoch_i}")
        
        

        #### -----------------------------------------VALIDATION --------------------------------------------#####

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

        with torch.no_grad():
            for i, sample in enumerate(valid_dataloader):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                # Move all tensors in the batch to the device
                sample = {key: tensor.to(device) for key, tensor in sample.items()}

                # Get the source and target sentences from the batch
                src_sentences = sample["input_ids"]
                tgt_sentences = sample["target_ids"]
                attention_mask = sample["attention_mask"]

                 # Access the original TransformerModel from the DataParallel wrapper
                generator2_train_dp = generator2_train.module if isinstance(generator2_train, torch.nn.DataParallel) else generator2_train

                generator2_train_out = generator2_train_dp(input_ids=src_sentences, attention_mask=attention_mask, decoder_input_ids=tgt_sentences , output_hidden_states=True, return_dict=True)
                fake_tgt_sentences_probs = F.log_softmax(generator2_train_out.logits, dim=-1)
                fake_tgt_sentences_probs = fake_tgt_sentences_probs.view(
                    -1, fake_tgt_sentences_probs.size(-1)
                )  # Shape: [batch_size * seq_len, vocab_size]
                tgt_sentences_flat = tgt_sentences.view(-1)  # Shape: [batch_size * seq_len]

                # Calculate generator loss
                logits_flat = generator2_train_out.logits.view(-1, generator2_train_out.logits.size(-1))  # Shape: [batch_size * seq_len, vocab_size]
                g_loss = g_criterion(logits_flat, tgt_sentences_flat)

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

                g2_token_embeds = generator2_train_dp.model.shared(fake_tgt_sentences)
                g1_token_embeds = generator2_train_dp.model.shared(fake_tgt_sentences_G1_pretrain_probs['input_ids'].to(device))

                print("shape of g2 token embeds: ", g2_token_embeds.shape)
                print("shape of g1 token embeds: ", g1_token_embeds.shape)

                g_cosine_loss = cosine_embedding_loss(g2_token_embeds, g1_token_embeds)
                g_kl_loss_reverse = kl_divergence_loss_reverse(g2_token_embeds, g1_token_embeds)
                g_kl_loss = kl_divergence_loss(g2_token_embeds, g1_token_embeds)

                total_g_loss = total_g_loss = config['total_g_loss']['g_loss']*g_loss + config['total_g_loss']['g_cosine_loss']*g_cosine_loss + config['total_g_loss']['g_kl_loss']*g_kl_loss
                
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
                
                def init_db_valid():
                    conn = None
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
                        if conn:
                            conn.close()

                def log_translation_db_valid(
                    epoch_i,
                    src_sentences_converted_logging_org,
                    tgt_sentences_converted_logging_org,
                    fake_tgt_sentences_converted_logging_G2_train,
                    fake_tgt_sentences_G1_pretrain_converted_logging,
                    fake_tgt_sentences_G1_pretrain_org_translated_sent,
                ):
                    conn = None
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
                        if conn:
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
                d_loss = config['d_loss']['real_loss']*real_loss + config['d_loss']['fake_loss']*fake_loss + config['d_loss']['fake_loss_pretrain']*fake_loss_pretrain
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

        torch.save(
                generator2_train,
                open(checkpoints_path + f"valid_checkpoint_generator_dill_format_at_{epoch_i}.pt", "wb"),
                pickle_module=dill,
            )
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
            torch.save(
                generator2_train,
                open(checkpoints_path + f"best_generator_dill_open_at_{epoch_i}.pt", "wb"),
                pickle_module=dill,
            )
            # Saving G and Tokenizer using save_pretrained
            generator2_train.save_pretrained(checkpoints_path + f"best_generator_save_pretrained_at_{epoch_i}")
            tokenizer.save_pretrained(checkpoints_path + f"best_generator_tokenizer_save_pretrained_at_{epoch_i}")
        
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
            torch.save(discriminator_cnn, checkpoints_path + f"best_discriminator_dill_direct_at_{epoch_i}.pt", pickle_module=dill)
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

