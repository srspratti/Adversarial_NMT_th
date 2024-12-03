# import datasets
from datasets import load_dataset
import evaluate
import os
import torch
import dill
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import argparse
import options
import logging
import random
import numpy as np
import torch
seed = 1234
# import parser
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

parser = argparse.ArgumentParser(description="Adversarial-NMT-BERT")

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

g_and_d_loss_checkpoint_config =[
    # With K.D losses - Either cosine or KL ; Without PreTrain loss included in D loss
    # {   "combination" : "G2_cos_D_baseline_3_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd",
    # "total_g_loss" : {"g_loss":0.0, "g_cosine_loss":1.00,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.10, "fake_loss":0.90, "fake_loss_pretrain":0.00} 
    # }
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
    # }
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
    # }
    #        {   "combination" : "G_0_25_75_cos_kl_75_25_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd",
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
    # Dec 1st - 2024
    #      {   "combination" : "G_0_25_75_cos_kl_25_50_25_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_80",
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
    # {   "combination" : "G_0_75_25_cos_kl_10_90_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_80",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":0.75,"g_kl_loss":0.25}, 
    # "d_loss" : {"real_loss":0.10, "fake_loss":0.90, "fake_loss_pretrain":0.00} 
    # },
    #     {   "combination" : "G_0_75_25_cos_kl_25_50_25_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_80",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":0.75,"g_kl_loss":0.25}, 
    # "d_loss" : {"real_loss":0.25, "fake_loss":0.50, "fake_loss_pretrain":0.25} 
    # }
    ## bs 80 
    #       {   "combination" : "G_0_25_75_cos_kl_25_50_25_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_80",
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
    # {   "combination" : "G_0_75_25_cos_kl_10_90_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_80",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":0.75,"g_kl_loss":0.25}, 
    # "d_loss" : {"real_loss":0.10, "fake_loss":0.90, "fake_loss_pretrain":0.00} 
    # },
    #     {   "combination" : "G_0_75_25_cos_kl_25_50_25_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_80",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":0.75,"g_kl_loss":0.25}, 
    # "d_loss" : {"real_loss":0.25, "fake_loss":0.50, "fake_loss_pretrain":0.25} 
    # }
    # {   "combination" : "G_0_75_25_cos_kl_10_90_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":0.75,"g_kl_loss":0.25}, 
    # "d_loss" : {"real_loss":0.10, "fake_loss":0.90, "fake_loss_pretrain":0.00} 
    # },
    #     {   "combination" : "G_0_75_25_cos_kl_25_50_25_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":0.75,"g_kl_loss":0.25}, 
    # "d_loss" : {"real_loss":0.25, "fake_loss":0.50, "fake_loss_pretrain":0.25} 
    # }
    #     {   "combination" : "G_1_0_0_cos_kl_0_0_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100",
    # "total_g_loss" : {"g_loss":1.00, "g_cosine_loss":0.00,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.00, "fake_loss":0.00, "fake_loss_pretrain":0.00} 
    # }
    #         {   "combination" : "G_1_0_0_cos_kl_0_1_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100",
    # "total_g_loss" : {"g_loss":1.00, "g_cosine_loss":0.00,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.00, "fake_loss":1.00, "fake_loss_pretrain":0.00} 
    # },
    # {   "combination" : "G_1_0_0_cos_kl_1_0_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100",
    # "total_g_loss" : {"g_loss":1.00, "g_cosine_loss":0.00,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":1.00, "fake_loss":0.00, "fake_loss_pretrain":0.00} 
    # },
    # {   "combination" : "G_1_0_0_cos_kl_0_0_1_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100",
    # "total_g_loss" : {"g_loss":1.00, "g_cosine_loss":0.00,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.00, "fake_loss":0.00, "fake_loss_pretrain":1.00} 
    # }
    # {   "combination" : "G_1_0_0_cos_kl_0_1_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100_10PG",
    # "total_g_loss" : {"g_loss":1.00, "g_cosine_loss":0.00,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.00, "fake_loss":1.00, "fake_loss_pretrain":0.00} 
    # }
    # {   "combination" : "G_1_0_0_cos_kl_0_1_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100_10PG_0001lr",
    # "total_g_loss" : {"g_loss":1.00, "g_cosine_loss":0.00,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.00, "fake_loss":1.00, "fake_loss_pretrain":0.00} 
    # }
    #     { "combination" : "G_0_50_50_cos_kl_10_90_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100_10PG_0001lr",
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
    # { "combination" : "G_10_90_0_cos_kl_10_90_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100_10PG_0001lr",
    # "total_g_loss" : {"g_loss":0.10, "g_cosine_loss":0.90,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.10, "fake_loss":0.90, "fake_loss_pretrain":0.00} 
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
    # { "combination" : "G_10_90_0_cos_kl_10_90_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100_100PG_00001lr",
    # "total_g_loss" : {"g_loss":0.10, "g_cosine_loss":0.90,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.10, "fake_loss":0.90, "fake_loss_pretrain":0.00} 
    # }
    # { "combination" : "G_25_75_0_cos_kl_10_90_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100_100PG_00001lr",
    # "total_g_loss" : {"g_loss":0.25, "g_cosine_loss":0.75,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.10, "fake_loss":0.90, "fake_loss_pretrain":0.00} 
    # }
    # { "combination" : "G_10_90_0_cos_kl_10_60_40_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100_100PG_00001lr",
    # "total_g_loss" : {"g_loss":0.05, "g_cosine_loss":0.95,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.10, "fake_loss":0.60, "fake_loss_pretrain":0.40} 
    # }
    #        { "combination" : "G_10_90_0_cos_kl_10_90_0_D_5000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100_100PG_00001lr",
    # "total_g_loss" : {"g_loss":0.10, "g_cosine_loss":0.90,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.10, "fake_loss":0.90, "fake_loss_pretrain":0.00} 
    # }
    #         { "combination" : "G_10_90_0_cos_kl_10_90_0_D_5000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100_1000PG_00001lr",
    # "total_g_loss" : {"g_loss":0.10, "g_cosine_loss":0.90,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.10, "fake_loss":0.90, "fake_loss_pretrain":0.00} 
    # }
    #                 { "combination" : "G_0_1_0_cos_kl_0_0_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100_0PG_0001lr",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":1.00,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.00, "fake_loss":0.00, "fake_loss_pretrain":0.00} 
    # },
    #         { "combination" : "G_0_0_1_cos_kl_0_0_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100_0PG_0001lr",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":0.00,"g_kl_loss":1.00}, 
    # "d_loss" : {"real_loss":0.00, "fake_loss":0.00, "fake_loss_pretrain":0.00} 
    # },
    #                 { "combination" : "G_0_50_50_cos_kl_0_0_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100_0PG_0001lr",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":0.50,"g_kl_loss":0.50}, 
    # "d_loss" : {"real_loss":0.00, "fake_loss":0.00, "fake_loss_pretrain":0.00} 
    # }
    # { "combination" : "G_0_1_0_cos_kl_10_100_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100_10PG_0001lr",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":1.00,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":1, "fake_loss":10, "fake_loss_pretrain":0.00} 
    # }
    # { "combination" : "G_0_1_0_cos_kl_10_100_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100_100PG_00001lr",
    # "total_g_loss" : {"g_loss":0.00, "g_cosine_loss":1.00,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":1, "fake_loss":10, "fake_loss_pretrain":0.00} 
    # }
    #     { "combination" : "G_10_100_0_cos_kl_10_90_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100_100PG_00001lr",
    # "total_g_loss" : {"g_loss":0.10, "g_cosine_loss":10.00,"g_kl_loss":0.00}, 
    # "d_loss" : {"real_loss":0.1, "fake_loss":0.9, "fake_loss_pretrain":0.00} 
    # }
    { "combination" : "G_10_500_0_cos_kl_10_90_0_D_1000_to_1_mil_only_biasTermsUpd_crl_upc_every_1_updates_PGloss_1_every_2_upd_bs_100_100PG_00001lr",
    "total_g_loss" : {"g_loss":0.10, "g_cosine_loss":50.00,"g_kl_loss":0.00}, 
    "d_loss" : {"real_loss":0.1, "fake_loss":0.9, "fake_loss_pretrain":0.00} 
    }
]


def main(args, config):

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
    
    
    getpwd = os.getcwd()
    bleu_metric = evaluate.load("sacrebleu")
    meteor_metric = evaluate.load("meteor")
    rouge_metric = evaluate.load("rouge")
    ter_metric = evaluate.load("ter")
    # chrf_metric = evaluate.load("chrf")
    # bleurt_metric = evaluate.load("bleurt")
    comet_metric = evaluate.load("comet")
    # gleu_metric = evaluate.load("gleu")
    # hlepor_metric = evaluate.load("hlepor")
    # bertscore_metric = evaluate.load("bertscore")

    if not os.path.exists(os.path.join(getpwd, "checkpoints", "bert_dualG", "g_and_d_loss_checkpoint_config_metrics_translations_output",config['combination'])):
        os.makedirs(os.path.join(getpwd, "checkpoints", "bert_dualG", "g_and_d_loss_checkpoint_config_metrics_translations_output",config['combination']))
        
    
    # checkpoint_path_generator = '/home/paperspace/google_drive_v1/Research_Thesis/2024/git_repo/checkpoints/bert_dualG/wmt14_en_fr_1mil_pg_kd_loss_MarianMT_unfreezeonlylmlayer_600sents_dedbug_spcChars__save_pretrained_v2/best_generator_at_1.pt'
    # checkpoint_path_tokenizer = "/home/paperspace/google_drive_v1/Research_Thesis/2024/git_repo/checkpoints/bert_dualG/wmt14_en_fr_1mil_pg_kd_loss_MarianMT_unfreezeonlylmlayer_600sents_dedbug_spcChars__save_pretrained_v2/best_generator_tokenizer_save_pretrained_at_1"
    # translations_generated_filename = "translated_french_by_MarianMT_FT_600sents.txt"
    
    checkpoint_path_generator = os.path.join(getpwd, "checkpoints", "bert_dualG", "wmt14_en_fr_1mil_pg_kd_loss_MarianMT_unfreezeonlylmlayer_debug_Normalkd_comb_" + config['combination'] +'_save_open_direct_pretrained'+'/train_checkpoint_generator_save_pretrained_at_1')
    checkpoint_path_tokenizer = os.path.join(getpwd, "checkpoints", "bert_dualG", "wmt14_en_fr_1mil_pg_kd_loss_MarianMT_unfreezeonlylmlayer_debug_Normalkd_comb_" + config['combination'] +'_save_open_direct_pretrained'+'/train_checkpoint_tokenizer_save_pretrained_at_1')

    # Load the entire model directly
    # generator2_checkpoint = torch.load(open(checkpoint_path_generator, "rb"), pickle_module=dill)
    generator2_checkpoint = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path_generator)

    # generator2_train # Extract the underlying model from the DataParallel wrapper
    generator2_checkpoint = generator2_checkpoint.module if isinstance(generator2_checkpoint, torch.nn.DataParallel) else generator2_checkpoint

    # Check if CUDA is available and then set the default device to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # generator2_checkpoint.eval()
    # generator2_checkpoint.to('device')
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path_tokenizer)

    batch_size = 8  # Adjust this based on your GPU's memory capacity
    translations= []

    if torch.cuda.device_count() > 1:
        generator2_checkpoint = torch.nn.DataParallel(generator2_checkpoint).cuda()
    else:
        generator2_checkpoint.cuda()

    generator2_checkpoint = generator2_checkpoint.module if hasattr(generator2_checkpoint, 'module') else generator2_checkpoint

    generator2_checkpoint.eval()

    # Assuming 'texts' is defined elsewhere and contains the English sentences to be translated
    # for idx, text in tqdm(enumerate(texts), desc="Translating", total=len(texts)):
    #     inputs = tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt").input_ids.to(device)
        
    #     # Generate the outputs using the model
    #     outputs = generator2_checkpoint.generate(inputs, max_length=60, num_beams=5, early_stopping=True)
        
    #     # Decode the generated IDs to text
    #     translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #     translations.append(translation)

    # # Save the translations to a text file
    # getpwd = os.getcwd()
    # file_path = os.path.join(getpwd, translations_generated_filename)

    # with open(file_path, "w") as file:
    #     for translation in translations:
    #         file.write(translation + "\n")

    for i in tqdm(range(0, len(texts), batch_size), desc="Translating batches"):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, truncation=True, padding="max_length", max_length=128, return_tensors="pt").input_ids.to(device)
        # print("inputs shape: ", inputs.shape)
        # Generate outputs for the entire batch
        outputs = generator2_checkpoint.generate(inputs, max_length=60, num_beams=5, early_stopping=True)
        # print("outputs shape", outputs.shape)

        # Decode all outputs in the batch
        # batch_translations = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        batch_translation = tokenizer.batch_decode(outputs , skip_special_tokens=True)
        translations.extend(batch_translation)

    # Save the translations to a text file - translations
    import os
    # os.path.join(getpwd, "checkpoints", "bert_dualG", "g_and_d_loss_checkpoint_config_metrics_translations_output",config['combination'])
    file_path = os.path.join(os.getcwd(), "checkpoints", "bert_dualG", "g_and_d_loss_checkpoint_config_metrics_translations_output",config['combination'],"translated_batch_"+config['combination']+".txt")
    with open(file_path, "w") as file:
        for translation in translations:
            file.write(translation + "\n")


    # result_batch = metric.compute(predictions=translations, references=labels)
    # result_batch = {"bleu": result_batch["score"]}
    # result_batch

    results = {
        "combination": config["combination"],
        "g_loss": config["total_g_loss"],
        "d_loss": config["d_loss"],
        "bleu": bleu_metric.compute(predictions=translations, references=labels)["score"],
        "meteor": meteor_metric.compute(predictions=translations, references=labels)["meteor"],
        "rouge": rouge_metric.compute(predictions=translations, references=labels),
        "ter": ter_metric.compute(predictions=translations, references=labels)["score"],
        # "chrf": chrf_metric.compute(predictions=translations, references=labels)["score"],
        # "bleurt": bleurt_metric.compute(predictions=translations, references=labels)["scores"],
        "comet": comet_metric.compute(predictions=translations, references=labels, sources=texts)["mean_score"]
        # "gleu": gleu_metric.compute(predictions=translations, references=labels)["score"],
        # "hlepor": hlepor_metric.compute(predictions=translations, references=labels)["score"],
        # "bertscore": bertscore_metric.compute(predictions=translations, references=labels, lang="en")
    }


    file_path_en = os.path.join(getpwd,"checkpoints", "bert_dualG", "g_and_d_loss_checkpoint_config_metrics_translations_output",config['combination'] ,"original_english_"+config['combination']+".txt")
    # file_path = "/path/to/translations.txt"

    # Open the file in write mode
    with open(file_path_en, "w") as file:
        # Write each translation to the file
        for text in texts:
            file.write(text + "\n")

    
    file_path_fr = os.path.join(getpwd,"checkpoints", "bert_dualG", "g_and_d_loss_checkpoint_config_metrics_translations_output",config['combination'] ,"original_french_"+config['combination']+".txt")
    # file_path = "/path/to/translations.txt"

    # Open the file in write mode
    with open(file_path_fr, "w") as file:
        # Write each translation to the file
        for label in labels:
            file.write(label + "\n")



     # Save results to a file
    results_file_path = os.path.join(getpwd,"checkpoints", "bert_dualG", "g_and_d_loss_checkpoint_config_metrics_translations_output",config['combination'] ,"results_" + config['combination'] + ".txt")
    with open(results_file_path, "w") as f:
        f.write("Combination: " + results["combination"] + "\n")
        f.write("G Loss: " + str(results["g_loss"]) + "\n")
        f.write("D Loss: " + str(results["d_loss"]) + "\n")
        f.write("BLEU Score: " + str(results["bleu"]) + "\n")
        f.write("METEOR Score: " + str(results["meteor"]) + "\n")
        f.write("ROUGE Scores: " + str(results["rouge"]) + "\n")
        f.write("TER Score: " + str(results["ter"]) + "\n")
        # f.write("CHR-F Score: " + str(results["chrf"]) + "\n")
        # f.write("BLEURT Scores: " + str(results["bleurt"]) + "\n")
        f.write("COMET Score: " + str(results["comet"]) + "\n")
        # f.write("GLEU Score: " + str(results["gleu"]) + "\n")
        # f.write("HLEPOR Score: " + str(results["hlepor"]) + "\n")
        # f.write("BERTScore: Precision: " + str(results["bertscore"]["precision"]) + "\n")
        # f.write("BERTScore: Recall: " + str(results["bertscore"]["recall"]) + "\n")
        # f.write("BERTScore: F1: " + str(results["bertscore"]["f1"]) + "\n")


if __name__ == "__main__":
    # main()
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        logging.warning(f"unknown arguments: {parser.parse_known_args()[1]}")
    for config in tqdm(g_and_d_loss_checkpoint_config, desc="Running models with diff. g_loss and d_loss"):
        print(" running config ", config["combination"])
        main(options, config)