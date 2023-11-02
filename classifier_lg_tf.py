import argparse
import logging
import re
import pandas as pd

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

from sequence_generator import SequenceGenerator
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix  # for metrics

import random
seed = random.randint(0, 2**32 - 1)
torch.manual_seed(569084360)
print("seed: ", 569084360)

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
    return re.sub(r'@@ ?', '', text)

def ids_to_sentences(src_tokens, dict):
    # Assuming src_tokens is a 2D tensor, 
    # convert to list of lists and then convert each list of ids to a sentence.
    
    # Ensure src_tokens is on the CPU and then convert to a list of lists
    src_tokens_list = src_tokens.cpu().numpy().tolist()
    
    sentences = []
    print("dict , ", dict)
    # print("dict , ", dict.__len__())
    for ids in src_tokens_list:
        # words = [dict.__getitem__(idx) for idx in ids if idx != dict.eos_index]
        words = [dict.__getitem__(idx) for idx in ids if idx not in [dict.eos_index, dict.pad_index]]
        sentence = ' '.join(words)
        # Remove padding from the sentences 
        # Un apply bpe to each sentence
        sentences.append(unbpe(sentence))    
    return sentences


def main(args):
    
    import gc
    torch.cuda.empty_cache()
    gc.collect()

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
    d_model_path = '/root/Adversarial_NMT_th/checkpoints/joint/test_vastai_wmt14_en_fr_2023_32mil_to_35mil_8mgpu_3070Dloss_fullDict_v3/train_joint_d_0.079.epoch_1.pt'
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
    g_model_path = '/root/Adversarial_NMT_th/checkpoints/joint/test_vastai_wmt14_en_fr_2023_32mil_to_35mil_8mgpu_3070Dloss_fullDict_v3/train_joint_g_10.570.epoch_1.pt'
    assert os.path.exists(g_model_path)
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
    
    
    if use_cuda:
        if torch.cuda.device_count() > 1:
            discriminator = torch.nn.DataParallel(discriminator).cuda()
            generator = torch.nn.DataParallel(generator).cuda()
        else:
            generator.cuda()
            discriminator.cuda()
    else:
        discriminator.cpu()
        generator.cpu()
    
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
        
        src_sentences_converted.extend(src_sentences_converted_temp)
        target_converted.extend(target_converted_temp)
        ht_mt_target_converted.extend(ht_mt_target_converted_temp)
        
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
        print("fake_sentence_generator_converted_temp: ", fake_sentence_generator_converted_temp)     
        fake_sentence_generator_converted.extend(fake_sentence_generator_converted_temp)
        
        
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
    
    print("y_true: ", len(y_true))
    print("y_pred: ", len(y_pred))
    print("src_sentences_converted: ", len(src_sentences_converted))
    print("target_converted: ", len(target_converted))
    print("ht_mt_target_converted: ", len(ht_mt_target_converted))
    print("fake_sentence_generator_converted ", len(fake_sentence_generator_converted))
    print("seed: ", seed)
    
    # data = {}
    classify_df = pd.DataFrame(data={"src_sentences_converted":src_sentences_converted, "target_converted": target_converted ,"ht_mt_target_converted": ht_mt_target_converted, "fake_sentence_generator_converted":fake_sentence_generator_converted, "y_true": y_true, "y_pred": y_pred})
    # classify_df =classify_df.transpose()
    classify_df['y_true'] = classify_df['y_true'].replace({1: "human", 0: "machine"})
    classify_df['y_pred'] = classify_df['y_pred'].replace({1: "human", 0: "machine"})

    
    # classify_df.to_excel('classify_df_8mil_o19_v2.xlsx', index=False) # 8 million 8 gpu with 70 30 loss
    # classify_df.to_excel('classify_df_8mil_o21_v1.xlsx', index=False)   # 8 million 8 gpu with 30 70 loss
    # classify_df.to_excel('classify_df_8mil_o22_v1.xlsx', index=False)   # 8 million 8 gpu with 30 70 loss
    classify_df.to_excel('classify_df_8mil_o24_v1.xlsx', index=False)   # 8 million 8 gpu with only D loss Only Fake/Machine
    

if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        logging.warning("unknown arguments: {0}".format(
            parser.parse_known_args()[1]))
    main(options)
