import argparse
import logging
import re
import pandas as pd
import html

import torch
import os
from torch import cuda
import options
import data
from generator import LSTMModel
from generator_tf_ptfseq import TransformerModel_custom
from discriminator import Discriminator
import utils
from tqdm import tqdm
from torch.autograd import Variable
from collections import OrderedDict
from meters import AverageMeter
from dictionary import Dictionary
from typing import Sequence

from textblob import TextBlob
from lexical_diversity import lex_div as ld

import spacy
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

import re
import subprocess
import pandas as pd
import nltk
nltk.download('wordnet')
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
nltk.download('punkt')

import pandas as pd
from sacrebleu import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge

# Load spaCy's language model for English and French
# nlp_en = spacy.load('en_core_web_sm')
# nlp_en = spacy.load('en_core_web_md')
nlp_en = spacy.load('en_core_web_lg')

# nlp_fr = spacy.load('fr_core_web_sm')
# nlp_fr = spacy.load('fr_core_web_md')
nlp_fr = spacy.load('fr_core_news_lg')

from sklearn.metrics.pairwise import cosine_similarity

from sequence_generator import SequenceGenerator
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix  # for metrics

import random
seed = random.randint(0, 2**32 - 1)
torch.manual_seed(1964997770)
# torch.manual_seed(seed)
print("seed used: ", 1964997770)

# torch.manual_seed(3203451255)
# torch.manual_seed(seed)
# print("seed: ", seed)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

parser = argparse.ArgumentParser(
    description="Driver program for JHU Adversarial-NMT.")

# parser = argparse.ArgumentParser(description="Adversarial-NMT.")

torch.cuda.empty_cache()
# Load args
options.add_general_args(parser)
options.add_dataset_args(parser)
options.add_distributed_training_args(parser)
options.add_optimization_args(parser)
options.add_checkpoint_args(parser)
options.add_generator_model_args(parser)
options.add_discriminator_model_args(parser)
options.add_generation_args(parser)

# Metric Calculation Functions    
def calculate_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average=None, zero_division=1)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)

    # Count occurrences for each class in y_true and y_pred
    true_machine_count = y_true.count(0)
    true_human_count = y_true.count(1)
    pred_machine_count = y_pred.count(0)
    pred_human_count = y_pred.count(1)
    
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    # Calculate accuracy for each class
    human_accuracy_cm = (TP + TN) / len(y_true)
    machine_accuracy_cm = (TN + FP) / len(y_true)
    overall_accuracy_cm = (TP + TN) / len(y_true)

    # Calculate accuracy for each class
    machine_accuracy = sum([1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0]) / true_machine_count
    human_accuracy = sum([1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1]) / true_human_count

    return human_accuracy_cm, machine_accuracy_cm, overall_accuracy_cm, precision, recall, f1, machine_accuracy, human_accuracy, true_machine_count, true_human_count, pred_machine_count, pred_human_count, TN, FP, FN, TP


def unbpe(text):
    """Un-apply BPE encoding from a text.
    
    Args:
        text (str): BPE-encoded text.
    
    Returns:
        str: Text with BPE encoding removed.
    """
        # Using regex to replace instances of "@@ " with an empty string
    print("before unbpe ", text)
    text_unbpe = re.sub(r'@@ ?', '', text)
    # return re.sub(r'@@ ?', '', text)
    print("after unbpe ", text_unbpe)
    return text_unbpe

dict_obj = Dictionary()
def ids_to_sentences(src_tokens, dict):
    # Assuming src_tokens is a 2D tensor, 
    # convert to list of lists and then convert each list of ids to a sentence.
    
    # Ensure src_tokens is on the CPU and then convert to a list of lists
    src_tokens_list = src_tokens.cpu().numpy().tolist()
    
    sentences = []
    print("dict , ", dict)
    print("dict , ", dict.__len__())
    for ids in src_tokens_list:
        # words = [dict.__getitem__(idx) for idx in ids if idx != dict.eos_index]
        print("dict.eos_index, ",dict.eos_index)
        print("dict.pad_index ",dict.pad_index)
        for idx in ids:
            print("idx: ", idx)
            print("dict.__getitem__(idx) ",dict.__getitem__(idx))
        words = [dict.__getitem__(idx) for idx in ids if idx not in [dict.eos_index, dict.pad_index]]
        print("words: ", words)
        sentence = ' '.join(words)
        # Remove padding from the sentences 
        # Un apply bpe to each sentence
        sentences.append(unbpe(sentence))    
    return sentences


def main(args):
    
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    
    # Replace with the path to your learned BPE codes.
    BPE_CODE_PATH = "/root/Adversarial_NMT_th/pretrained_models/wmt14.en-fr.joined-dict.transformer/code"
    # /u/prattisr/phase-2/all_repos/Adversarial_NMT/neural-machine-translation-using-gan-master/subword-nmt/subword_nmt/apply_bpe.py
    def apply_bpe(sentence):
        """Apply BPE encoding to a sentence using a subprocess call to apply_bpe.py."""
        # Construct the shell command to execute apply_bpe.py.
        # cmd = f"echo '{sentence}' | python /path/to/apply_bpe.py -c {BPE_CODE_PATH}"
        cmd = f"echo '{sentence}' | python /root/Adversarial_NMT_th/subword-nmt/subword_nmt/apply_bpe.py -c {BPE_CODE_PATH}"
        # Execute the command and get the output.
        bpe_sentence = subprocess.check_output(cmd, shell=True, text=True).strip()
        return bpe_sentence

    # torch.cuda.empty_cache()
    print("args.fixed_max_len) ", args.fixed_max_len)
    # use_cuda = (len(args.gpuid) >= 1)
    use_cuda = True
    dataset = data.load_raw_text_dataset_test_classify(
                args.data,
                ['test'],
                args.src_lang,
                args.trg_lang,
                args.fixed_max_len,
            )
    if args.src_lang is None or args.trg_lang is None:
            # record inferred languages in args, so that it's saved in checkpoints
            args.src_lang, args.trg_lang = dataset.src, dataset.dst

    print("type of dataset: ", type(dataset))
    print('| [{}] dictionary: {} types'.format(
        dataset.src, len(dataset.src_dict)))
    print('| [{}] dictionary: {} types'.format(
        dataset.dst, len(dataset.dst_dict)))
    print('| {} {} {} examples'.format(
        args.data, 'test', len(dataset.splits['test'])))

    d_logging_meters = OrderedDict()
    # d_logging_meters['train_loss'] = AverageMeter()
    d_logging_meters['test_classify_loss'] = AverageMeter()
    # d_logging_meters['train_acc'] = AverageMeter()
    d_logging_meters['test_classify_acc'] = AverageMeter()
    d_logging_meters['bsz'] = AverageMeter()  # sentences per batch
    
    # Set model parameters
    args.encoder_embed_dim = 1000
    args.fixed_max_len = 50
    args.decoder_embed_dim = 1000
    
    args.encoder_layers = 2 # 4
    args.encoder_dropout_out = 0
    
    args.encoder_heads = 2
    args.encoder_ffn_embed_dim = 1000
    
    args.decoder_heads = 2
    args.decoder_ffn_embed_dim = 1000
    
    args.decoder_layers = 2 # 4
    args.decoder_out_embed_dim = 1000
    args.decoder_dropout_out = 0

    
    ########### --- Loading best discriminator ---------###################
    
    # d_model_path = 'checkpoints/joint/wmt14_en_fr_raw_sm_v2/best_dmodel_at_best_gmodel.pt' # LSTM 
    import os
    # d_model_path = os.getcwd()+'/'+'checkpoints/joint/wmt14_en_fr_raw_sm_tf_v2/test_best_dmodel_at_best_gmodel.pt' # TF
    # d_model_path = 'checkpoints/joint/test_wmt14_en_fr_raw_sm_v2/test_best_dmodel_at_best_gmodel.pt'
    # d_model_path = '/u/prattisr/phase-2/all_repos/Adversarial_NMT/neural-machine-translation-using-gan-master/checkpoints/joint/test_wmt14_en_fr_raw_sm_tf_v2/test_best_dmodel_at_best_gmodel.pt'
    # d_model_path = '/u/prattisr/phase-2/all_repos/Adversarial_NMT/neural-machine-translation-using-gan-master/checkpoints/joint/test_wmt14_en_fr_raw_sm_tf_v3/tf_best_dmodel_at_best_gmodel.pt'
    # d_model_path = '/u/prattisr/phase-2/all_repos/Adversarial_NMT/neural-machine-translation-using-gan-master/checkpoints/joint/test_wmt14_en_fr_raw_sm_tf_disc_v3/tf_disc_best_dmodel_at_best_gmodel.pt'
    # d_model_path = '/u/prattisr/phase-2/all_repos/Adversarial_NMT/neural-machine-translation-using-gan-master/checkpoints/joint/test_wmt14_en_fr_raw_sm_tf_disc_v3/test_joint_d_0.691.epoch_25.pt'
    # d_model_path = '/u/prattisr/phase-2/all_repos/checkpoints/joint/test_wmt14_en_fr_raw_sm_tf_disc7030_s20_v1/tf_disc_best_dmodel_at_best_gmodel.pt'
    # test_wmt14_en_fr_2023_pt_oc5_sm_50k_v1
    # d_model_path='/u/prattisr/phase-2/all_repos/Adversarial_NMT/neural-machine-translation-using-gan-master/pretrained_models/checkpoints/joint/test_wmt14_en_fr_2023_pt_oc5_sm_50k_v1/tf_disc_best_dmodel_at_best_gmodel.pt'
    # d_model_path='/u/prattisr/phase-2/all_repos/Adversarial_NMT/neural-machine-translation-using-gan-master/checkpoints/joint/test_vastai_wmt14_en_fr_2023_8mil_8mgpu_v2/train_joint_d_0.017.epoch_1.pt' #  8 million 8gpu 70 30 loss v2
    # d_model_path = '/u/prattisr/phase-2/all_repos/Adversarial_NMT/neural-machine-translation-using-gan-master/checkpoints/joint/test_vastai_wmt14_en_fr_2023_8mil_8mgpu_5050_v2/train_joint_d_0.030.epoch_1.pt' # 8 million 8gpu 50 50 loss v2
    # d_model_path = '/u/prattisr/phase-2/all_repos/Adversarial_NMT/neural-machine-translation-using-gan-master/checkpoints/joint/test_vastai_wmt14_en_fr_2023_8mil_8mgpu_3070_v2/train_joint_d_0.026.epoch_1.pt' # 8 million 8gpu 30 70 loss v2
    # d_model_path = 'checkpoints/joint/test_vastai_wmt14_en_fr_2023_24mil_to_32mil_8mgpu_3070Dloss_fullDict_v3/train_joint_d_0.018.epoch_1.pt'
    # d_model_path = '/root/Adversarial_NMT_th/checkpoints/joint/test_vastai_wmt14_en_fr_2023_8mil_8mgpu_3070Dloss_fullDict_v3/train_joint_d_0.048.epoch_1.pt'
    # d_model_path = '/root/Adversarial_NMT_th/checkpoints/joint/test_vastai_wmt14_en_fr_2023_32mil_to_35mil_8mgpu_3070Dloss_fullDict_v3/train_joint_d_0.079.epoch_1.pt'
    
    # d_model_path = '/root/Adversarial_NMT_th/pretrained_models/checkpoints/joint/test_vastai_wmt14_en_fr_2023_4mil_8mgpu_3070Dloss_4milDict_v1/train_joint_d_0.036.epoch_1.pt'
    # d_model_path = '/root/Adversarial_NMT_th/checkpoints/joint/test_wmt14_en_fr_2023_40mil__mgpu_ptfseqOnly_v1/train_joint_d_0.056.epoch_1.pt'
    # d_model_path = '/root/Adversarial_NMT_th/checkpoints/joint/test_wmt14_en_fr_2023_40mil__mgpu_ptfseqOnly_v2_20_40_40/train_joint_d_0.027.epoch_1.pt'
    # d_model_path = '/root/Adversarial_NMT_th/checkpoints/joint/test_wmt14_en_fr_2023_40mil__mgpu_ptfseqOnly_v2_0_50_50/train_joint_d_0.028.epoch_1.pt'
    # d_model_path = '/root/Adversarial_NMT_th/checkpoints/joint/test_wmt14_en_fr_2023_40mil__mgpu_ptfseqOnly_v2_0_80_20/train_joint_d_0.021.epoch_6.pt'
    d_model_path = '/root/Adversarial_NMT_th/checkpoints/joint/test_wmt14_en_fr_2023_40mil__mgpu_ptfseqOnly_v2_0_100_0/train_joint_d_0.001.epoch_1.pt'
    print("d_model_path ", d_model_path)
    
    assert os.path.exists(d_model_path)
    discriminator = Discriminator(args, dataset.src_dict, dataset.dst_dict,  use_cuda=use_cuda)
    model_dict_dmodel = discriminator.state_dict()
    model_dmodel = torch.load(d_model_path)
    pretrained_dict_dmodel = model_dmodel.state_dict()
    print("model defined type :\n ", type(model_dict_dmodel))
    # print("model defined :\n ", model_dict_dmodel)
    print("model pretrained type :\n ", type(pretrained_dict_dmodel))
    # print("model pretrained :\n ", pretrained_dict_dmodel)
    
    # filter out unnecessary keys
    pretrained_dict_dmodel = {k:v for k,v in pretrained_dict_dmodel.items() if k in model_dict_dmodel}
    
    # 2. overwrite entries in the existing state dict
    model_dict_dmodel.update(pretrained_dict_dmodel)
    
    #3. load the new state dict
    discriminator.load_state_dict(model_dict_dmodel)
    
    # fix discriminator word embedding (as Wu et al. do)
    for p in discriminator.embed_src_tokens.parameters():
        p.requires_grad = False
    for p in discriminator.embed_trg_tokens.parameters():
        p.requires_grad = False
    
    discriminator.eval()
    
    print("Best Discriminator loaded successfully!")
    
    d_criterion = torch.nn.BCELoss()
    
     ### Loading pre-trained generator for analysing the BLEU score and perplexity 
    # g_model_path = 'checkpoints/joint/test_vastai_wmt14_en_fr_2023_24mil_to_32mil_8mgpu_3070Dloss_fullDict_v3/train_joint_g_10.497.epoch_1.pt'
    # g_model_path = '/root/Adversarial_NMT_th/checkpoints/joint/test_vastai_wmt14_en_fr_2023_8mil_8mgpu_3070Dloss_fullDict_v3/train_joint_g_10.352.epoch_1.pt'
    # g_model_path = '/root/Adversarial_NMT_th/checkpoints/joint/test_vastai_wmt14_en_fr_2023_32mil_to_35mil_8mgpu_3070Dloss_fullDict_v3/train_joint_g_10.570.epoch_1.pt'
    
    # g_model_path = '/root/Adversarial_NMT_th/pretrained_models/checkpoints/joint/test_vastai_wmt14_en_fr_2023_4mil_8mgpu_3070Dloss_4milDict_v1/train_joint_g_10.401.epoch_1.pt'
    # g_model_path = '/root/Adversarial_NMT_th/checkpoints/joint/test_wmt14_en_fr_2023_40mil__mgpu_ptfseqOnly_v1/train_joint_g_10.395.epoch_1.pt'
    # g_model_path = '/root/Adversarial_NMT_th/checkpoints/joint/test_wmt14_en_fr_2023_40mil__mgpu_ptfseqOnly_v2_20_40_40/train_joint_g_10.400.epoch_1.pt'
    # g_model_path = '/root/Adversarial_NMT_th/checkpoints/joint/test_wmt14_en_fr_2023_40mil__mgpu_ptfseqOnly_v2_0_50_50/train_joint_g_10.399.epoch_1.pt'
    # g_model_path = '/root/Adversarial_NMT_th/checkpoints/joint/test_wmt14_en_fr_2023_40mil__mgpu_ptfseqOnly_v2_0_80_20/train_joint_g_10.339.epoch_6.pt'
    g_model_path = '/root/Adversarial_NMT_th/checkpoints/joint/test_wmt14_en_fr_2023_40mil__mgpu_ptfseqOnly_v2_0_100_0/train_joint_g_10.398.epoch_1.pt'
    assert os.path.exists(g_model_path)
    print("g_model_path: ", g_model_path)
     # Load model
    # g_model_path = 'checkpoints/joint/best_gmodel.pt'
    # g_model_path = '/u/prattisr/phase-2/all_repos/Adversarial_NMT/neural-machine-translation-using-gan-master/pretrained_models/checkpoints/joint/test_wmt14_en_fr_2023_pt_oc5_sm_50k_v1/test_disc_best_gmodel.pt'
    # g_model_path='/u/prattisr/phase-2/all_repos/Adversarial_NMT/neural-machine-translation-using-gan-master/pretrained_models/checkpoints/joint/test_wmt14_en_fr_2023_pt_oc5_sm_50k_v1/tf_disc_best_gmodel.pt'
    # g_model_path='/u/prattisr/phase-2/all_repos/Adversarial_NMT/neural-machine-translation-using-gan-master/checkpoints/joint/test_vastai_wmt14_en_fr_2023_8mil_8mgpu_v2/train_joint_g_10.353.epoch_1.pt' # 8 million 8gpu v2
    # g_model_path = '/u/prattisr/phase-2/all_repos/Adversarial_NMT/neural-machine-translation-using-gan-master/checkpoints/joint/test_vastai_wmt14_en_fr_2023_8mil_8mgpu_5050_v2/train_joint_g_10.346.epoch_1.pt' # 8 million 50 50 8gpu v2
    # g_model_path = '/u/prattisr/phase-2/all_repos/Adversarial_NMT/neural-machine-translation-using-gan-master/checkpoints/joint/test_vastai_wmt14_en_fr_2023_8mil_8mgpu_3070_v2/train_joint_g_10.355.epoch_1.pt'  # 8 million 30 70 8gpu v2
    
    
    generator = TransformerModel_custom(args, 
                dataset.src_dict, dataset.dst_dict, use_cuda=use_cuda)    
    model_dict = generator.state_dict()
    model = torch.load(g_model_path)
    pretrained_dict = model.state_dict()
    print("pretrained_dict type: ", type(pretrained_dict))
    print("model : ",model)
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k,
                       v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    generator.load_state_dict(model_dict)
    generator.eval()

    print("Generator loaded successfully!")
    
    
    ################ Loading Pre-Trained Generator ( FairSeq PreTrained Tranformer) #####################
    
    path_to_your_pretrained_model = '/root/Adversarial_NMT_th/pretrained_models/wmt14.en-fr.joined-dict.transformer'
    from fairseq.models.transformer import TransformerModel
    generator_pt = TransformerModel.from_pretrained(
    model_name_or_path=path_to_your_pretrained_model,
    checkpoint_file='model.pt',
    bpe='subword_nmt',
    # data_name_or_path='/u/prattisr/phase-2/all_repos/Adversarial_NMT/neural-machine-translation-using-gan-master/data-bin/wmt14_en_fr_raw_sm/50kLines',
    data_name_or_path='/root/Adversarial_NMT_th/pretrained_models/wmt14.en-fr.joined-dict.transformer',
    bpe_codes = '/root/Adversarial_NMT_th/pretrained_models/wmt14.en-fr.joined-dict.transformer/bpecodes'
    )
    print("Pretrained Generator loaded successfully!")
    
    if use_cuda:
        if torch.cuda.device_count() > 1:
            discriminator = torch.nn.DataParallel(discriminator).cuda()
            generator = torch.nn.DataParallel(generator).cuda()
            generator_pt = torch.nn.DataParallel(generator_pt).cuda()
        else:
            generator.cuda()
            discriminator.cuda()
            generator_pt.cuda()
    else:
        discriminator.cpu()
        generator.cpu()
        generator_pt.cpu()
    
    max_positions = int(1e5)
    
    
    # initialize dataloader
    max_positions_test = (args.fixed_max_len, args.fixed_max_len)
    print("max_positions_test",max_positions_test)
    testloader = dataset.eval_dataloader_test_classify(
    'test',
    max_tokens=args.max_tokens,
    max_sentences=args.joint_batch_size,
    max_positions=max_positions_test,
    skip_invalid_size_inputs_valid_test=True,
    # sample_without_replacement=args.sample_without_replacement,
    descending=False,  # largest batch first to warm the caching allocator
    shard_id=args.distributed_rank,
    num_shards=args.distributed_world_size
    )
  
    # discriminator validation
    
    y_true = []
    y_pred = []
    src_sentences_converted = []
    target_converted = []
    ht_mt_target_converted = []
    fake_sentence_generator_converted = []
    fake_sentence_generator_converted_dict = []
    translations_all= []
    
    print("type of testloader:", type(testloader))
    for batch in testloader:
        print("type of batch:",type(batch))

    print("len of testloader:", len(testloader))
    print("testloader.dataset:", testloader.dataset)
    # print("type of testloader:", type(testloader))
    
    for i, sample in tqdm(enumerate(testloader)):
        
        print("i ", i)
        print("sample before use_cuda: ", sample)
        if use_cuda:
            sample = utils.make_variable(sample, cuda=cuda)
        print("sample after use_cuda: ", sample)
        print("sample size() after use_cuda: ", sample.keys())
        print("sample['id']: ",sample['id'])
        print("in testloader")
            
        bsz = sample['target'].size(0)
        src_sentence = sample['net_input']['src_tokens'] # En
        print("src_sentence size ", src_sentence.size())
        print("src_sentence  ", src_sentence)
        target = sample['target'] # Fr Human Translated
        print("target size ", target.size())
        print("target  ", target)
        ht_mt_target = sample['ht_mt_target_trans']['ht_mt_target'] # Fr Human or Machine
        print("ht_mt_target size ", ht_mt_target.size())
        print("ht_mt_target  ", ht_mt_target)
        ht_mt_label = sample['ht_mt_target_trans']['ht_mt_label'] # Fr human or Machine labels - 1 for human and 0 for Machine
        print("ht_mt_label size ", ht_mt_label.size())
        print("ht_mt_label  ", ht_mt_label)
        
        print("This {} is the bsz type".format(type(bsz)))
        print("This {} is the src_sentence type".format(type(src_sentence)))
        print("This {} is the target type".format(type(target)))
        print("This {} is the ht_mt_target type".format(type(ht_mt_target)))
        print("This {} is the ht_mt_label type".format(type(ht_mt_label)))

    
        # disc_out = discriminator(src_sentToBeTranslated, hm_or_mch_translSent)
        disc_out = discriminator(src_sentence, ht_mt_target)
        # If disc_out is 1 -> Human else if disc_out is 0 -> Machine
        print("disc_out ", disc_out)
        d_loss = d_criterion(disc_out.squeeze(1), ht_mt_label.float())
        print("d_loss ", d_loss)
        # acc = torch.sum(torch.round(disc_out).squeeze(1) == ht_mt_label).float() / len(ht_mt_label)
        # print("acc: ", acc)
        
        # Short-cut : Converting the ids to sentences 
        
        src_sentences_converted_temp = ids_to_sentences(src_sentence, dataset.src_dict)
        target_converted_temp = ids_to_sentences(target, dataset.dst_dict)
        ht_mt_target_converted_temp = ids_to_sentences(ht_mt_target, dataset.dst_dict)
        
        # src_sentences_converted_temp = dict_obj.ids_to_sentences(src_sentence)
        # target_converted_temp = dict_obj.ids_to_sentences(target)
        # ht_mt_target_converted_temp = dict_obj.ids_to_sentences(ht_mt_target)
        
        print("src_sentences_converted_temp: ",src_sentences_converted_temp)
        src_sentences_converted.extend(src_sentences_converted_temp)
        target_converted.extend(target_converted_temp)
        ht_mt_target_converted.extend(ht_mt_target_converted_temp)
        
        ####### Pre-Trained Generator Transformer Analysis ##################
        
        # Access the original TransformerModel from the DataParallel wrapper
        original_generator_pt = generator_pt.module if isinstance(generator_pt, torch.nn.DataParallel) else generator_pt

        # Now use the translate method
        print("src_sentences_converted: ", src_sentences_converted)
        translations_temp = original_generator_pt.translate(src_sentences_converted_temp)
        print("translations : ", translations_temp)
        translations_all.extend(translations_temp)
        
        ###### Generator sample for further analysis 
        
        with torch.no_grad():
            sys_out_batch = generator(sample=sample, args=args)

        out_batch = sys_out_batch.contiguous().view(-1, sys_out_batch.size(-1)) # (64 X 50) X 6632  

        _,prediction = out_batch.topk(1)
        print("prediction before squeeze: ",prediction.size())
        prediction = prediction.squeeze(1)  #64 * 50 = 6632
        print("prediction after squeeze: ",prediction.size())
        fake_sentence = torch.reshape(prediction, src_sentence.shape) # 64 X 50 
        
        print("fake_sentence ", fake_sentence)

        print("fake_sentence after torch.reshape(prediction, src_sentence.shape) : ",fake_sentence.size())
        
        fake_sentence_generator_converted_temp = ids_to_sentences(fake_sentence, dataset.dst_dict)   
        # fake_sentence_generator_converted_temp = dict_obj.ids_to_sentences(fake_sentence)
        print("fake_sentence_generator_converted_temp: ", fake_sentence_generator_converted_temp)     
        fake_sentence_generator_converted.extend(fake_sentence_generator_converted_temp)
        
        # print("def string from the Dictionary")
        
        # fake_sentence_generator_converted_temp_dict = dict_obj.string(fake_sentence, bpe_symbol='@@')
        # print("fake_sentence_generator_converted_temp_dict: ", fake_sentence_generator_converted_temp_dict)     
        # fake_sentence_generator_converted_dict.extend(fake_sentence_generator_converted_temp_dict)
         
        #############
        disc_out_rounded = torch.round(disc_out).squeeze(1)
        print("ht_mt_label ", ht_mt_label.tolist())
        print("disc_out_rounded ", disc_out_rounded.tolist())
        y_true.extend(ht_mt_label.tolist())
        y_pred.extend(disc_out_rounded.tolist())
        acc = torch.sum(disc_out_rounded == ht_mt_label.float())/ len(ht_mt_label)
        print("acc: ", acc)
        print("y_true: ", y_true)
        print("y_pred: ", y_pred)
        
        d_logging_meters['test_classify_acc'].update(acc)
        d_logging_meters['test_classify_loss'].update(d_loss)
        logging.debug(f"D test_classify loss {d_logging_meters['test_classify_loss'].avg:.3f}, acc {d_logging_meters['test_classify_acc'].avg:.3f} at batch {i}")
        # torch.cuda.empty_cache()
    
    #######################################################################################
    
    # Once all samples are evaluated, calculate overall metrics
    # precision, recall, f1, machine_accuracy, human_accuracy, true_machine_count, true_human_count, pred_machine_count, pred_human_count = calculate_metrics(y_true, y_pred)

    
    #######################################################################################
    human_accuracy_cm, machine_accuracy_cm, overall_accuracy_cm, precision, recall, f1, machine_accuracy, human_accuracy, true_machine_count, true_human_count, pred_machine_count, pred_human_count, TN, FP, FN, TP = calculate_metrics(y_true, y_pred)
    
    print("Confusion Matrix:")
    print(f"TN: {TN} | FP: {FP}")
    print(f"FN: {FN} | TP: {TP}\n")

    print("Machine Translated Metrics (Negative Class):*******************")
    print(f"True Negatives: {TN}")
    print(f"False Positives: {FP}")
    
    print(f"Accuracy machine_accuracy_cm : {machine_accuracy_cm:.3f}\n")
    
    print(f"True Machine Translated Count: {true_machine_count}")
    print(f"Predicted Machine Translated Count: {pred_machine_count}")
    
    print(f"Machine Translated Precision: {precision[0]:.3f}")
    print(f"Machine Translated Recall: {recall[0]:.3f}")
    print(f"Machine Translated F1 Score: {f1[0]:.3f}")
    
    print(f"Machine Translated Accuracy: {machine_accuracy:.3f}")

    print("Human Translated Metrics (Positive Class):********************")
    print(f"True Positives: {TP}")
    print(f"False Negatives: {FN}")
    # print(f"Precision: {precision[1]:.3f}")
    # print(f"Recall: {recall[1]:.3f}")
    # print(f"F1 Score: {f1[1]:.3f}")
    
    print(f"Accuracy human_accuracy_cm : {human_accuracy_cm:.3f}\n")
    
    print(f"True Human Translated Count: {true_human_count}")
    print(f"Predicted Human Translated Count: {pred_human_count}")
    
    print(f"Human Translated Precision: {precision[1]:.3f}")
    print(f"Human Translated Recall: {recall[1]:.3f}")
    print(f"Human Translated F1 Score: {f1[1]:.3f}")
    
    print(f"Human Translated Accuracy: {human_accuracy:.3f}")

    print(f"Overall Accuracy overall_accuracy_cm : {overall_accuracy_cm:.3f}")
    
    ######################################################################################

    print("y_true: ", y_true)
    y_pred = [int(y_pre) for y_pre in y_pred]
    print("y_pred: ", y_pred)
    print("src_sentences_converted: ", src_sentences_converted)
    print("target_converted: ", target_converted)
    print("ht_mt_target_converted: ", ht_mt_target_converted)
    print("fake_sentence_generator_converted ", fake_sentence_generator_converted)
    print("translations_all: ", translations_all)
    # print("fake_sentence_generator_converted_dict ", fake_sentence_generator_converted_dict)
    
    print("y_true: ", len(y_true))
    print("y_pred: ", len(y_pred))
    print("src_sentences_converted: ", len(src_sentences_converted))
    print("target_converted: ", len(target_converted))
    print("ht_mt_target_converted: ", len(ht_mt_target_converted))
    print("fake_sentence_generator_converted ", len(fake_sentence_generator_converted))
    print("len of translations_all ", len(translations_all))
    print("seed: ", seed)
    
    # data = {}
    classify_df = pd.DataFrame(data={"src_sentences_converted":src_sentences_converted, "target_converted": target_converted ,"ht_mt_target_converted": ht_mt_target_converted, "fake_sentence_generator_converted":fake_sentence_generator_converted, "Translated_by_FairSeqTransf_G":translations_all,"y_true": y_true, "y_pred": y_pred})
    # classify_df =classify_df.transpose()
    classify_df['y_true'] = classify_df['y_true'].replace({1: "human", 0: "machine"})
    classify_df['y_pred'] = classify_df['y_pred'].replace({1: "human", 0: "machine"})

    
    # classify_df.to_excel('classify_df_8mil_o19_v2.xlsx', index=False) # 8 million 8 gpu with 70 30 loss
    # classify_df.to_excel('classify_df_8mil_o21_v1.xlsx', index=False)   # 8 million 8 gpu with 30 70 loss
    # classify_df.to_excel('classify_df_8mil_o22_v1.xlsx', index=False)   # 8 million 8 gpu with 30 70 loss
    # classify_df.to_excel('classify_df_8mil_D3_v1.xlsx', index=False)   # 8 million 8 gpu with only D loss Only Fake/Machine
    # classify_df.to_excel('classify_df_4mil_D10_v1.xlsx', index=False) 
    # classify_df.to_excel('classify_df_4mil_D16_40mil_ptfseqOnly_v2_0_100_0_v1.xlsx', index=False)
    classify_df.to_excel('classify_df_4mil_D29_4mil_ptfseqOnly_v2_0_100_0_all_Metrics_seed_1964997770_60_v8-1.xlsx', index=False) 
    
    # def calculate_bleu(hypotheses, references):
    #     return corpus_bleu(hypotheses, [references]).score
     
    def calculate_bleu(hypotheses: Sequence[str], references: Sequence[Sequence[str]]) -> float:
        """
        Calculate the BLEU score for hypotheses against references.

        Args:
            hypotheses (Sequence[str]): A sequence of hypothesis sentences.
            references (Sequence[Sequence[str]]): A sequence where each element is a 
                                                    sequence of reference sentences.

        Returns:
            float: The BLEU score, or 0 if there are no valid hypotheses and references.
        """
        # if not hypotheses or not references:
        # if hypotheses.empty or references.empty:
        #     # No valid data to calculate BLEU score
        #     return 0.0

        # formatted_hypotheses = [str(hyp) if hyp is not None else '' for hyp in hypotheses]
        # formatted_references = [[str(ref) if ref is not None else ''] for ref in references]
        
        formatted_hypotheses = [str(hyp) if not pd.isna(hyp) else '' for hyp in hypotheses]
        formatted_references = [[str(ref) if not pd.isna(ref) else ''] for ref in references]

        print("formatted_hypotheses ",formatted_hypotheses)
        print("formatted_references ",formatted_references)
        
        # Using NLTK's smoothing function
        # chencherry = SmoothingFunction()
        
        # return corpus_bleu(formatted_hypotheses, formatted_references, smoothing_function=chencherry.method1)
        return corpus_bleu(formatted_hypotheses, formatted_references).score

    def calculate_bleu_smoothing(hypotheses: Sequence[str], references: Sequence[Sequence[str]]) -> float:
        
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        
        """
        Calculate the BLEU score for hypotheses against references.

        Args:
            hypotheses (Sequence[str]): A sequence of hypothesis sentences.
            references (Sequence[Sequence[str]]): A sequence where each element is a 
                                                    sequence of reference sentences.

        Returns:
            float: The BLEU score, or 0 if there are no valid hypotheses and references.
        """
        # Formatting references for corpus-level BLEU score calculation
        # Each reference should be a list of tokenized sentences
        # formatted_hypotheses = machine_translations['ht_mt_target_converted']
        # formatted_references = machine_translations['target_converted']
        
        formatted_hypotheses = hypotheses
        formatted_references = references
        
        formatted_references_corpus = [[ref[0].split()] for ref in formatted_references]

        # Formatting hypotheses for corpus-level BLEU score calculation
        # Each hypothesis should be a tokenized sentence
        formatted_hypotheses_corpus = [hyp.split() for hyp in formatted_hypotheses]

        # Calculating corpus-level BLEU score
        # corpus_bleu_score = corpus_bleu(formatted_references_corpus, formatted_hypotheses_corpus)
            
        # Using NLTK's smoothing function
        chencherry = SmoothingFunction()
        
        return corpus_bleu(formatted_references_corpus, formatted_hypotheses_corpus, smoothing_function=chencherry.method1)
        # return corpus_bleu(formatted_hypotheses, formatted_references).score


    # # Formatting references for corpus-level BLEU score calculation
    # # Each reference should be a list of tokenized sentences
    # formatted_references_corpus = [[ref[0].split()] for ref in formatted_references]

    # # Formatting hypotheses for corpus-level BLEU score calculation
    # # Each hypothesis should be a tokenized sentence
    # formatted_hypotheses_corpus = [hyp.split() for hyp in formatted_hypotheses]

    # # Calculating corpus-level BLEU score
    # corpus_bleu_score = corpus_bleu(formatted_references_corpus, formatted_hypotheses_corpus)
    # corpus_bleu_score


    
    def calculate_bleu_old(hypotheses: Sequence[str], references: Sequence[Sequence[str]]) -> float:
        """
        Calculate the BLEU score for hypotheses against references.

        Args:
            hypotheses (Sequence[str]): A sequence of hypothesis sentences.
            references (Sequence[Sequence[str]]): A sequence where each element is a 
                                                sequence of reference sentences.

        Returns:
            float: The BLEU score.
        """
        # Ensure that each hypothesis is a string
        formatted_hypotheses = [str(hyp) if hyp is not None else '' for hyp in hypotheses]

        # Ensure that each reference is a list of strings
        formatted_references = [[str(ref) if ref is not None else ''] for ref in references]

        return corpus_bleu(formatted_hypotheses, formatted_references).score

    # def calculate_meteor(hypotheses, references):
    #     return sum(meteor_score([ref], hyp) for hyp, ref in zip(hypotheses, references)) / len(hypotheses)
    
    def calculate_meteor(hypotheses: Sequence[str], references: Sequence[str]) -> float:
        """
        Calculate the METEOR score for hypotheses against references.

        Args:
            hypotheses (Sequence[str]): A sequence of hypothesis sentences.
            references (Sequence[str]): A sequence of reference sentences.

        Returns:
            float: The METEOR score.
        """
        tokenized_hypotheses = [word_tokenize(str(hyp)) for hyp in hypotheses]
        tokenized_references = [word_tokenize(str(ref)) for ref in references]

        scores = [meteor_score([ref], hyp) for hyp, ref in zip(tokenized_hypotheses, tokenized_references)]
        return sum(scores) / len(scores) if scores else 0.0


    # def calculate_rouge(hypotheses, references):
    #     rouge = Rouge()
    #     scores = rouge.get_scores(hypotheses, references, avg=True)
    #     return scores  # This returns a dictionary with various ROUGE scores
    
    def calculate_rouge(hypotheses: Sequence[str], references: Sequence[str]) -> list:
        """
        Calculate the ROUGE score for each pair of hypothesis and reference sentences.

        Args:
            hypotheses (Sequence[str]): A sequence of hypothesis sentences.
            references (Sequence[str]): A sequence of reference sentences.

        Returns:
            list: A list of ROUGE scores for each hypothesis-reference pair.
        """
        rouge = Rouge()
        scores = rouge.get_scores(hypotheses, references)

        # Extract a specific ROUGE score for each pair, e.g., ROUGE-1
        rouge_scores = [score['rouge-1']['f'] for score in scores]  # Example for ROUGE-1 F1 score
        return rouge_scores

    # BLEU Score for 'fake_sentence_generator_converted' and 'Translated_by_FairSeqTransf_G'
    # classify_df['bleu_fake_gen'] = calculate_bleu(classify_df['fake_sentence_generator_converted'], classify_df['target_converted'])
    # classify_df['bleu_fairseq_gen'] = calculate_bleu(classify_df['Translated_by_FairSeqTransf_G'], classify_df['target_converted'])
    
    classify_df['bleu_fake_gen'] = calculate_bleu(classify_df['fake_sentence_generator_converted'].tolist(), 
                                              classify_df['target_converted'].apply(lambda x: [x]).tolist())
    classify_df['bleu_fairseq_gen'] = calculate_bleu(classify_df['Translated_by_FairSeqTransf_G'].tolist(), 
                                              classify_df['target_converted'].apply(lambda x: [x]).tolist())

    # METEOR and ROUGE for 'fake_sentence_generator_converted' and 'Translated_by_FairSeqTransf_G'
    # classify_df['meteor_fake_gen'] = calculate_meteor(classify_df['fake_sentence_generator_converted'], classify_df['target_converted'])
    # classify_df['meteor_fairseq_gen'] = calculate_meteor(classify_df['Translated_by_FairSeqTransf_G'], classify_df['target_converted'])
    
    classify_df['meteor_fake_gen'] = calculate_meteor(classify_df['fake_sentence_generator_converted'].tolist(), 
                                                  classify_df['target_converted'].tolist())

    classify_df['meteor_fairseq_gen'] = calculate_meteor(classify_df['Translated_by_FairSeqTransf_G'].tolist(), 
                                                  classify_df['target_converted'].tolist())


    classify_df['rouge_fake_gen'] = calculate_rouge(classify_df['fake_sentence_generator_converted'].tolist(), classify_df['target_converted'].tolist())
    classify_df['rouge_fairseq_gen'] = calculate_rouge(classify_df['Translated_by_FairSeqTransf_G'].tolist(), classify_df['target_converted'].tolist())

    print(classify_df['y_true'].dtypes)
    print(classify_df['y_true'].unique())

    
    # Calculate BLEU, METEOR, and ROUGE for machine-translated sentences only in 'ht_mt_target_converted'
    # mask = classify_df['y_true'] == "0"  # 0 indicates machine translation
    mask = classify_df['y_true'].astype(str) == 'machine'  # Convert to string if necessary
    machine_translations = classify_df[mask]
    
    
    print("mask", mask)
    print("machine_translations: ", machine_translations)
    
    # # When calculating BLEU for machine translations
    # mask = classify_df['y_true'] == 0  # 0 indicates machine translation
    # machine_translations = classify_df[mask]

    # Ensure that there are valid entries before calculation
    # if not machine_translations.empty:
    #     bleu_score = calculate_bleu(machine_translations['ht_mt_target_converted'].dropna().tolist(), 
    #                                 machine_translations['target_converted'].dropna().apply(lambda x: [x]).tolist())
    #     classify_df.loc[mask, 'bleu_machine_translations'] = bleu_score
    # else:
    #     classify_df.loc[mask, 'bleu_machine_translations'] = 0.0
    
    print("machine_translations['ht_mt_target_converted'], len:", len(machine_translations['ht_mt_target_converted']))
    print("machine_translations['ht_mt_target_converted'], ", machine_translations['ht_mt_target_converted'])
    print("machine_translations['target_converted'], len:", len(machine_translations['target_converted']))
    print("machine_translations['target_converted'], ", machine_translations['target_converted'])
    
    # bleu_machine_translations = calculate_bleu(machine_translations['ht_mt_target_converted'], machine_translations['target_converted'])
    # print("calculate_bleu(machine_translations['ht_mt_target_converted'], machine_translations['target_converted']) ", bleu_machine_translations)
    
    # Formatting references for corpus-level BLEU score calculation
    # Each reference should be a list of tokenized sentences
    # formatted_hypotheses = machine_translations['ht_mt_target_converted']
    # formatted_references = machine_translations['target_converted']
    
    # formatted_references_corpus = [[ref[0].split()] for ref in formatted_references]

    # # Formatting hypotheses for corpus-level BLEU score calculation
    # # Each hypothesis should be a tokenized sentence
    # formatted_hypotheses_corpus = [hyp.split() for hyp in formatted_hypotheses]

    # Calculating corpus-level BLEU score
    corpus_bleu_score = calculate_bleu_smoothing(machine_translations['ht_mt_target_converted'], machine_translations['target_converted'])
    # corpus_bleu_score
    
    # classify_df.loc[mask, 'bleu_machine_translations'] = calculate_bleu(machine_translations['ht_mt_target_converted'], machine_translations['target_converted'])
    classify_df.loc[mask, 'bleu_machine_translations'] = corpus_bleu_score
    
    classify_df.loc[mask, 'meteor_machine_translations'] = calculate_meteor(machine_translations['ht_mt_target_converted'], machine_translations['target_converted'])
    classify_df.loc[mask, 'rouge_machine_translations'] = calculate_rouge(machine_translations['ht_mt_target_converted'], machine_translations['target_converted'])

    print("classify_df.loc[mask, 'bleu_machine_translations']  ", classify_df['bleu_machine_translations'])
    
    
    #### Adding new columns - bleu_machine_translations_all , which calculates the score without any mask ( mask = classify_df['y_true'].astype(str) == 'machine' )) 
    classify_df['bleu_machine_translations_all'] = calculate_bleu(classify_df['ht_mt_target_converted'].tolist(), 
                                              classify_df['target_converted'].apply(lambda x: [x]).tolist())
    
    
    """
    classify_df['src_sentences_converted'] = classify_df['src_sentences_converted'].apply(html.unescape)
    classify_df['target_converted'] = classify_df['target_converted'].apply(html.unescape)
    classify_df['ht_mt_target_converted'] = classify_df['ht_mt_target_converted'].apply(html.unescape)
    classify_df['fake_sentence_generator_converted'] = classify_df['fake_sentence_generator_converted'].apply(html.unescape)
    classify_df['Translated_by_FairSeqTransf_G'] = classify_df['Translated_by_FairSeqTransf_G'].apply(html.unescape)
    """
    
    def clean_text(text):
        text = html.unescape(text)  # Decode HTML entities
        text = text.replace('@-@', '')  # Remove '@-@'
        return text
    
    classify_df['src_sentences_converted'] = classify_df['src_sentences_converted'].apply(clean_text)
    classify_df['target_converted'] = classify_df['target_converted'].apply(clean_text)
    classify_df['ht_mt_target_converted'] = classify_df['ht_mt_target_converted'].apply(clean_text)
    classify_df['fake_sentence_generator_converted'] = classify_df['fake_sentence_generator_converted'].apply(clean_text)
    classify_df['Translated_by_FairSeqTransf_G'] = classify_df['Translated_by_FairSeqTransf_G'].apply(clean_text)
    
    classify_df.to_excel('classify_df_4mil_D29_4mil_ptfseqOnly_v2_0_100_0_all_Metrics_seed_1964997770_60_v8-2.xlsx', index=False) 
    
    # Load a pre-trained sentence transformer model
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

    # Function to calculate semantic similarity
    def semantic_similarity_ht_machine(source_text, translated_text):
        embeddings = model.encode([source_text, translated_text])
        return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    # Function to calculate syntactic similarity
    # def syntactic_similarity(source_text, translated_text):
    #     doc_source = nlp_en(source_text)
    #     doc_translated = nlp_fr(translated_text)
    #     return doc_source.similarity(doc_translated)
    
    # Function to calculate semantic similarity using sentence embeddings
    def semantic_similarity(source_text, translated_text):
        embeddings = model.encode([source_text, translated_text])
        return cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1))[0][0]

    # Function to analyze syntactic similarity using dependency parsing
    def syntactic_similarity(source_text, translated_text):
        doc_source = nlp_en(source_text)
        doc_translated = nlp_fr(translated_text)
        # Compare the dependency trees of the two documents (this is a placeholder, real implementation would be more complex)
        return doc_source.similarity(doc_translated)

    # Iterate over the DataFrame and apply the analysis functions
    for index, row in classify_df.iterrows():
        source_text = row['src_sentences_converted']
        human_translated_text = row['target_converted']
        preTrG_translated_text = row['Translated_by_FairSeqTransf_G']

        # Semantic similarity
        classify_df.at[index, 'preTrG_Semantic_Similarity'] = semantic_similarity(source_text, preTrG_translated_text)
        classify_df.at[index, 'human_Semantic_Similarity'] = semantic_similarity(source_text, human_translated_text)
        # classify_df.loc[mask, 'ht_mt_target_Semantic_Similarity'] = semantic_similarity(source_text, ht_mt_target_converted)

        # Syntactic similarity
        classify_df.at[index, 'preTrG_Syntactic_Similarity'] = syntactic_similarity(source_text, preTrG_translated_text)
        classify_df.at[index, 'human_Syntactic_Similarity'] = syntactic_similarity(source_text, human_translated_text)
        # classify_df.loc[mask, 'ht_mt_target_Syntactic_Similarity'] = syntactic_similarity(source_text, ht_mt_target_converted)
    
    # Iterate over the DataFrame
    for index, row in classify_df.iterrows():
        # if mask[index]:  # Check if the current row is specified by the mask
        source_text = row['src_sentences_converted']
        ht_mt_target_converted = row['ht_mt_target_converted']

        # Calculate semantic similarity
        sem_sim = semantic_similarity_ht_machine(source_text, ht_mt_target_converted)
        classify_df.at[index, 'ht_mt_target_Semantic_Similarity'] = sem_sim

        # Calculate syntactic similarity
        syn_sim = syntactic_similarity(source_text, ht_mt_target_converted)
        classify_df.at[index, 'ht_mt_target_Syntactic_Similarity'] = syn_sim

    # Display the updated DataFrame
    # print(classify_df)
    classify_df.to_excel('classify_df_4mil_D29_4mil_ptfseqOnly_v2_0_100_0_all_Metrics_seed_1964997770_60_v8-3.xlsx', index=False) 
    
    from collections import Counter
    # from nltk.tokenize import word_tokenize
    
    def calculate_yules_i(text):
        tokens = word_tokenize(text.lower())
        token_counter = Counter(tokens)
        m1 = sum(token_counter.values())
        m2 = sum([freq ** 2 for freq in token_counter.values()])
        yules_i = 10 ** 4 * (m2 - m1) / (m1 ** 2) if m1 > 1 else 0
        return yules_i
    
    # Function to calculate Type/Token Ratio (TTR)
    def calculate_ttr(text):
        tokens = text.split()
        types = set(tokens)
        return len(types) / len(tokens)

    # Function to calculate Yule's I
    def calculate_yules_i_sm(text):
        return ld.yule(text)

    # Function to calculate MTLD
    def calculate_mtld(text):
        return ld.mtld(text.split())

    # Function for LIP Analysis (example: Sentiment Analysis)
    def analyze_sentiment(text):
        return TextBlob(text).sentiment.polarity

    # Apply calculations
    classify_df['target_converted_ttr'] = classify_df['target_converted'].apply(calculate_ttr) # target_converted ht_mt_target_converted Translated_by_FairSeqTransf_G
    classify_df['ht_mt_target_converted_ttr'] = classify_df['ht_mt_target_converted'].apply(calculate_ttr)
    classify_df['Translated_by_FairSeqTransf_G_ttr'] = classify_df['Translated_by_FairSeqTransf_G'].apply(calculate_ttr)
    
    classify_df['target_converted_yules_i'] = classify_df['target_converted'].apply(calculate_yules_i) # target_converted ht_mt_target_converted Translated_by_FairSeqTransf_G
    classify_df['ht_mt_target_converted_yules_i'] = classify_df['ht_mt_target_converted'].apply(calculate_yules_i)
    classify_df['Translated_by_FairSeqTransf_G_yules_i'] = classify_df['Translated_by_FairSeqTransf_G'].apply(calculate_yules_i)
    
    classify_df['target_converted_mtld'] = classify_df['target_converted'].apply(calculate_mtld) # target_converted ht_mt_target_converted Translated_by_FairSeqTransf_G
    classify_df['ht_mt_target_converted_mtld'] = classify_df['ht_mt_target_converted'].apply(calculate_mtld)
    classify_df['Translated_by_FairSeqTransf_G_mtld'] = classify_df['Translated_by_FairSeqTransf_G'].apply(calculate_mtld)
    
    classify_df['target_converted_sent'] = classify_df['target_converted'].apply(analyze_sentiment) # target_converted ht_mt_target_converted Translated_by_FairSeqTransf_G
    classify_df['ht_mt_target_converted_sent'] = classify_df['ht_mt_target_converted'].apply(analyze_sentiment)
    classify_df['Translated_by_FairSeqTransf_G_sent'] = classify_df['Translated_by_FairSeqTransf_G'].apply(analyze_sentiment)
    
    classify_df.to_excel('classify_df_4mil_D29_4mil_ptfseqOnly_v2_0_100_0_all_Metrics_seed_1964997770_60_v8-4.xlsx', index=False)
    
    # for col in classify_df.columns:
    #     classify_df[col + '_TTR'] = df[col].apply(calculate_ttr)
    #     classify_df[col + '_YulesI'] = df[col].apply(calculate_yules_i)
    #     classify_df[col + '_MTLD'] = df[col].apply(calculate_mtld)
    #     classify_df[col + '_Sentiment'] = df[col].apply(analyze_sentiment)


    
if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        logging.warning("unknown arguments: {0}".format(
            parser.parse_known_args()[1]))
    main(options)
