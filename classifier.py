import argparse
import logging

import torch
import os
from torch import cuda
import options
import data
from generator import LSTMModel
from discriminator import Discriminator
import utils
from tqdm import tqdm
from torch.autograd import Variable
from collections import OrderedDict
from meters import AverageMeter

from sequence_generator import SequenceGenerator
from sklearn.metrics import precision_score, recall_score, f1_score  # for metrics


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
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    y_true = [float(i) for i in y_true]
    print("y_pred ", y_pred)
    print("y_true ", y_true)
    y_pred = torch.tensor(y_pred)
    y_true = torch.tensor(y_true)
    accuracy = torch.sum((y_pred == y_true).float()) / y_true.numel()
    return precision, recall, f1, accuracy

def main(args):
    
    import gc
    torch.cuda.empty_cache()
    gc.collect()

    # torch.cuda.empty_cache()
    print("args.fixed_max_len) ", args.fixed_max_len)
    use_cuda = (len(args.gpuid) >= 1)
    if args.gpuid:
        cuda.set_device(args.gpuid[0])

        # Load dataset
        if args.replace_unk is not None:
            dataset = data.load_dataset(
                args.data,
                ['test'],
                args.src_lang,
                args.trg_lang,
            )
        else:
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
    d_model_path='/u/prattisr/phase-2/all_repos/Adversarial_NMT/neural-machine-translation-using-gan-master/pretrained_models/checkpoints/joint/test_wmt14_en_fr_2023_pt_oc5_sm_50k_v1/tf_disc_best_dmodel_at_best_gmodel.pt'
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
    
    if use_cuda > 0:
        discriminator.cuda()
    else:
        discriminator.cpu()
    
    max_positions = int(1e5)
    
    # initialize dataloader
    max_positions_test = (args.fixed_max_len, args.fixed_max_len)
    testloader = dataset.eval_dataloader_test_classify(
    'test',
    max_tokens=args.max_tokens,
    max_sentences=args.joint_batch_size,
    max_positions=max_positions_test,
    skip_invalid_size_inputs_valid_test=True,
    descending=True,  # largest batch first to warm the caching allocator
    shard_id=args.distributed_rank,
    num_shards=args.distributed_world_size
    )
  
    # discriminator validation
    
    y_true = []
    y_pred = []
    
    for i, sample in tqdm(enumerate(testloader)):
        
        print("i ", i)
        print("sample before use_cuda: ", sample)
        if use_cuda:
            sample = utils.make_variable(sample, cuda=cuda)
        print("sample after use_cuda: ", sample)
        print("sample size() after use_cuda: ", sample.keys())
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
        
        #
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
        
    # Once all samples are evaluated, calculate overall metrics
    precision, recall, f1, accuracy = calculate_metrics(y_true, y_pred)
    print(f"Overall Precision: {precision:.3f}")
    print(f"Overall Recall: {recall:.3f}")
    print(f"Overall F1 Score: {f1:.3f}")
    print(f"Overall Accuracy: {accuracy:.3f}")

    # Machine Translated sentences
    # pip install deep_translator
    # import deep_translator
    # from deep_translator import deepltranslator 
    # machineTranslSents = 
    
    # disc_out = discriminator(src_sentToBeTranslated, hm_or_mch_translSent)
    # If disc_out is 1 -> Human else if disc_out is 0 -> Machine
    
    """
    with torch.no_grad():
                    sys_out_batch = generator(sample)

                out_batch = sys_out_batch.contiguous().view(-1, sys_out_batch.size(-1)) # (64 X 50) X 6632  

                _,prediction = out_batch.topk(1)
                prediction = prediction.squeeze(1)  #64 * 50 = 6632

                fake_labels = Variable(torch.zeros(sample['target'].size(0)).float())

                fake_sentence = torch.reshape(prediction, src_sentence.shape) # 64 X 50 

                if use_cuda:
                    fake_labels = fake_labels.cuda()

                disc_out = discriminator(src_sentence, fake_sentence)
                d_loss = d_criterion(disc_out.squeeze(1), fake_labels)
                acc = torch.sum(torch.round(disc_out).squeeze(1) == fake_labels).float() / len(fake_labels)
                d_logging_meters['valid_acc'].update(acc)
                d_logging_meters['valid_loss'].update(d_loss)
                logging.debug(f"D dev loss {d_logging_meters['valid_loss'].avg:.3f}, acc {d_logging_meters['valid_acc'].avg:.3f} at batch {i}")
    """
    """
    translator = SequenceDiscriminator(
    generator, beam_size=args.beam, stop_early=(not args.no_early_stop),
    normalize_scores=(not args.unnormalized), len_penalty=args.lenpen,
    unk_penalty=args.unkpen)

    if use_cuda:
        translator.cuda()

    with torch.no_grad():
                    sys_out_batch = generator(sample)

                out_batch = sys_out_batch.contiguous().view(-1, sys_out_batch.size(-1)) # (64 X 50) X 6632  

                _,prediction = out_batch.topk(1)
                prediction = prediction.squeeze(1)  #64 * 50 = 6632

                fake_labels = Variable(torch.zeros(sample['target'].size(0)).float())

                fake_sentence = torch.reshape(prediction, src_sentence.shape) # 64 X 50 

                if use_cuda:
                    fake_labels = fake_labels.cuda()

                disc_out = discriminator(src_sentence, fake_sentence)
                d_loss = d_criterion(disc_out.squeeze(1), fake_labels)
                
    """
    #################### Writing predictions to file - If Required ###############
    """
    with open('predictions.txt', 'wb') as translation_writer:
        with open('real.txt', 'wb') as ground_truth_writer:

            translations = translator.generate_batched_itr(
                testloader, maxlen_a=args.max_len_a, maxlen_b=args.max_len_b, cuda=use_cuda)

            for sample_id, src_tokens, target_tokens, hypos in translations:
                # Process input and ground truth
                target_tokens = target_tokens.int().cpu()
                src_str = dataset.src_dict.string(src_tokens, args.remove_bpe)
                target_str = dataset.dst_dict.string(
                    target_tokens, args.remove_bpe, escape_unk=True)

                # Process top predictions
                for i, hypo in enumerate(hypos[:min(len(hypos), args.nbest)]):
                    hypo_tokens = hypo['tokens'].int().cpu()
                    hypo_str = dataset.dst_dict.string(
                        hypo_tokens, args.remove_bpe)

                    hypo_str += '\n'
                    target_str += '\n'

                    translation_writer.write(hypo_str.encode('utf-8'))
                    ground_truth_writer.write(target_str.encode('utf-8'))

"""

if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        logging.warning("unknown arguments: {0}".format(
            parser.parse_known_args()[1]))
    main(options)
