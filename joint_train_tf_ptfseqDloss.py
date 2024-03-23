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
getpwd = os.getcwd()
# sys.path.append(
#     "/u/prattisr/phase-2/all_repos/Adversarial_NMT/neural-machine-translation-using-gan-master"
# )
sys.path.append(getpwd)
# sys.path.append("/u/prattisr/phase-2/all_repos/Adversarial_NMT/neural-machine-translation-using-gan-master")
# https://stackoverflow.com/questions/67311527/how-to-set-gpu-count-to-0-using-os-environcuda-visible-devices
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7" 
"""
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
torch.cuda.device_count() # result is 2

os.environ["CUDA_VISIBLE_DEVICES"]="0"
torch.cuda.device_count() # result is 1, using first GPU

os.environ["CUDA_VISIBLE_DEVICES"]="1"
torch.cuda.device_count() # result is 1, using second GPU"""
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch import cuda
from torch.autograd import Variable

import data
import utils
from meters import AverageMeter
from discriminator import Discriminator
from generator_tf_ptfseq import TransformerModel_custom
from train_generator_new import train_g
from train_discriminator import train_d
from PGLoss import PGLoss
from tqdm import tqdm
from sequence_generator import SequenceGenerator
from dictionary import Dictionary

import re
import subprocess

import cProfile

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Adversarial-NMT.")

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
    # use_cuda = (len(args.gpuid) >= 1)
    # print("{0} GPU(s) are available".format(cuda.device_count()))
    # print("args.fixed_max_len) ", args.fixed_max_len)
    use_cuda = True
    print("{0} GPU(s) are available".format(cuda.device_count()))
    print("args.fixed_max_len) ", args.fixed_max_len)
    
    # Load dataset
    splits = ['train', 'valid']
    if data.has_binary_files(args.data, splits):
        dataset = data.load_dataset(
            args.data, splits, args.src_lang, args.trg_lang, args.fixed_max_len)
    else:
        dataset = data.load_raw_text_dataset(
            args.data, splits, args.src_lang, args.trg_lang, args.fixed_max_len)
    if args.src_lang is None or args.trg_lang is None:
        # record inferred languages in args, so that it's saved in checkpoints
        args.src_lang, args.trg_lang = dataset.src, dataset.dst

    print('| [{}] dictionary: {} types'.format(dataset.src, len(dataset.src_dict)))
    print('| [{}] dictionary: {} types'.format(dataset.dst, len(dataset.dst_dict)))
    
    for split in splits:
        print('| {} {} {} examples'.format(args.data, split, len(dataset.splits[split])))
        print('| type of dataset: ' , type(dataset.splits[split]))
        print('| actual dataset: ' , dataset.splits[split])
    
    g_logging_meters = OrderedDict()
    g_logging_meters['train_loss'] = AverageMeter()
    g_logging_meters['valid_loss'] = AverageMeter()
    g_logging_meters['train_acc'] = AverageMeter()
    g_logging_meters['valid_acc'] = AverageMeter()
    g_logging_meters['bsz'] = AverageMeter()  # sentences per batch

    d_logging_meters = OrderedDict()
    d_logging_meters['train_loss'] = AverageMeter()
    d_logging_meters['valid_loss'] = AverageMeter()
    d_logging_meters['train_acc'] = AverageMeter()
    d_logging_meters['valid_acc'] = AverageMeter()
    d_logging_meters['bsz'] = AverageMeter()  # sentences per batch

    # Set model parameters
    args.encoder_embed_dim = 1000
    args.encoder_layers = 2 # 4
    args.encoder_dropout_out = 0
    args.decoder_embed_dim = 1000
    
    args.encoder_heads = 2
    args.encoder_ffn_embed_dim = 1000
    
    args.decoder_heads = 2
    args.decoder_ffn_embed_dim = 1000
    
    args.decoder_layers = 2 # 4
    args.decoder_out_embed_dim = 1000
    args.decoder_dropout_out = 0
    args.bidirectional = False

    generator = TransformerModel_custom(args, dataset.src_dict, dataset.dst_dict, use_cuda=use_cuda)
    print("Generator TransformerModel loaded successfully!")
    discriminator = Discriminator(args, dataset.src_dict, dataset.dst_dict, use_cuda=use_cuda)
    print("Discriminator loaded successfully!")
    
    getpwd = os.getcwd()
     
    path_to_your_pretrained_model = getpwd + '/pretrained_models/wmt14.en-fr.joined-dict.transformer'
    from fairseq.models.transformer import TransformerModel
    generator_pt = TransformerModel.from_pretrained(
    model_name_or_path=path_to_your_pretrained_model,
    checkpoint_file='model.pt',
    bpe='subword_nmt',
    # data_name_or_path='/u/prattisr/phase-2/all_repos/Adversarial_NMT/neural-machine-translation-using-gan-master/data-bin/wmt14_en_fr_raw_sm/50kLines',
    data_name_or_path= getpwd + '/pretrained_models/wmt14.en-fr.joined-dict.transformer',
    bpe_codes = getpwd + '/pretrained_models/wmt14.en-fr.joined-dict.transformer/bpecodes'
    )
    print("Pretrained Generator loaded successfully!")
    
    def pad_sentences(sentences, pad_token='<pad>', max_len=None):
        if max_len is None:
            max_len = max(len(s.split()) for s in sentences)

        padded_sentences = []
        for sentence in sentences:
            words = sentence.split()
            num_padding = max_len - len(words)
            padded_sentence = ' '.join(words + [pad_token] * num_padding)
            padded_sentences.append(padded_sentence)
        
        return padded_sentences
        
    def unbpe(text):
        """Un-apply BPE encoding from a text.
        
        Args:
            text (str): BPE-encoded text.
        
        Returns:
            str: Text with BPE encoding removed.
        """
            # Using regex to replace instances of "@@ " with an empty string
        return re.sub(r'@@ ?', '', text)
    
    def ids_to_words(ids, dict):
        words = [dict.get_symbol(idx) for idx in ids if idx != dict.eos_index]
        return ' '.join(words)
    
    
    # converting sentences to ids 'out_batch'
    # import torch

    def sentences_to_ids_old(padded_bpe_translations, dict, max_len=None):
        # Determine the maximum length if not provided
        if max_len is None:
            max_len = max(len(sentence.split()) for sentence in padded_bpe_translations)
        
        # Initialize an empty list to store lists of token ids
        all_ids = []
        print("length of padded_bpe_translations:", len(padded_bpe_translations))
        for idx, sentence in enumerate(padded_bpe_translations):
            print("sentence ",sentence)
            words = sentence.split()
            # Convert words to ids, handling the padding
            ids = [dict.index(word) if word in dict else dict.unk_index for word in words]  # Handle unknown words
            # ids += [dict.pad_index] * (max_len - len(ids))  # Pad to max_len
            print("idx in sentences_to_ids:",idx)
            print(" length of ids: ", len(ids))
            all_ids.append(ids)
        
        # Convert list of lists into a 2D tensor
        print("all_ids ", all_ids)
        return torch.tensor(all_ids, dtype=torch.long)
    
    
    def sentences_to_ids_nondict(padded_bpe_translations, dict_obj, max_len=None):
        # Determine the maximum length if not provided
        if max_len is None:
            max_len = max(len(sentence.split()) for sentence in padded_bpe_translations)

        # Pre-allocate the tensor
        all_ids = torch.empty((len(padded_bpe_translations), max_len), dtype=torch.long)
        all_ids.fill_(dict_obj.pad_index)  # Assuming dict has a pad_index attribute

        print(" dictionary object: length", dict_obj.__len__)
        word_to_id_map = {word: dict_obj.index(word) for word in dict_obj}
        print(" word_to_id_map : ", word_to_id_map)
        
        for idx, sentence in enumerate(padded_bpe_translations):
            print("idx in sentences_to_ids:",idx)
            words = sentence.split()
            # ids = [dict.get(word, dict.unk_index) for word in words]  # More efficient lookup
            ids = [word_to_id_map.get(word, dict_obj.unk_index) for word in words]
            # ids = [dict_obj.index(word) if word in dict_obj else dict_obj.unk_index for word in words]
            all_ids[idx, :len(ids)] = torch.tensor(ids, dtype=torch.long)
              
        return all_ids
    
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
    
    # Replace with the path to your learned BPE codes.
    getpwd = os.getcwd()
    BPE_CODE_PATH = getpwd + "/pretrained_models/wmt14.en-fr.joined-dict.transformer/bpecodes"
    # /u/prattisr/phase-2/all_repos/Adversarial_NMT/neural-machine-translation-using-gan-master/subword-nmt/subword_nmt/apply_bpe.py
    def apply_bpe(sentence):
        """Apply BPE encoding to a sentence using a subprocess call to apply_bpe.py."""
        # Construct the shell command to execute apply_bpe.py.
        # cmd = f"echo '{sentence}' | python /path/to/apply_bpe.py -c {BPE_CODE_PATH}"
        cmd = f"echo '{sentence}' | python {getpwd}/subword-nmt/subword_nmt/apply_bpe.py -c {BPE_CODE_PATH}"
        # Execute the command and get the output.
        bpe_sentence = subprocess.check_output(cmd, shell=True, text=True).strip()
        return bpe_sentence
    
    # Chnages from vastai_114_8gpu    
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

    # adversarial training checkpoints saving path # test_wmt14_en_fr_2024_1mil_sgpu_ptfseqOnly_v2_0_100_0_btch_50_epch_20
    if not os.path.exists('checkpoints/joint/test_wmt14_en_fr_2024_1mil_gpu_ptfseqOnly_v2_0_100_0_btch_50_epch_20_ignore'):
        os.makedirs('checkpoints/joint/test_wmt14_en_fr_2024_1mil_gpu_ptfseqOnly_v2_0_100_0_btch_50_epch_20_ignore')
    checkpoints_path = 'checkpoints/joint/test_wmt14_en_fr_2024_1mil_gpu_ptfseqOnly_v2_0_100_0_btch_50_epch_20_ignore/'

    # define loss function
    g_criterion = torch.nn.NLLLoss(ignore_index=dataset.dst_dict.pad(),reduction='sum')
    d_criterion = torch.nn.BCELoss()
    pg_criterion = PGLoss(ignore_index=dataset.dst_dict.pad(), size_average=True,reduce=True)

    # fix discriminator word embedding (as Wu et al. do)
    
    if torch.cuda.device_count() > 1:
        for p in discriminator.module.embed_src_tokens.parameters():
            p.requires_grad = False
        for p in discriminator.module.embed_trg_tokens.parameters():
            p.requires_grad = False
    else:          
        for p in discriminator.embed_src_tokens.parameters():
            p.requires_grad = False
        for p in discriminator.embed_trg_tokens.parameters():
            p.requires_grad = False

    # define optimizer
    g_optimizer = eval("torch.optim." + args.g_optimizer)(filter(lambda x: x.requires_grad,
                                                                 generator.parameters()),
                                                          args.g_learning_rate)

    d_optimizer = eval("torch.optim." + args.d_optimizer)(filter(lambda x: x.requires_grad,
                                                                 discriminator.parameters()),
                                                          args.d_learning_rate,
                                                          momentum=args.momentum,
                                                          nesterov=True)

################################################################## EPOCHS #################################################################
    # start joint training
    best_dev_loss = math.inf
    num_update = 0
    # main training loop
    for epoch_i in tqdm(range(1, args.epochs + 1)):
        logging.info("At {0}-th epoch.".format(epoch_i))

        seed = args.seed + epoch_i
        torch.manual_seed(seed)

        max_positions_train = (args.fixed_max_len, args.fixed_max_len)
        
################################################################## TRAINING #################################################################
        
        # Initialize dataloader, starting at batch_offset
        trainloader = dataset.train_dataloader(
            'train',
            max_tokens=args.max_tokens,
            max_sentences=args.joint_batch_size,
            max_positions=max_positions_train,
            # seed=seed,
            epoch=epoch_i,
            sample_without_replacement=args.sample_without_replacement,
            sort_by_source_size=(epoch_i <= args.curriculum),
            shard_id=args.distributed_rank,
            num_shards=args.distributed_world_size,
        )
        
        
        # Custom train_loader
        # When calling train_dataloader(), specify the desired fixed batch size directly
        # fixed_batch_size = 80
        # trainloader = dataset.train_dataloader(
        #     'train',
        #     batch_size=fixed_batch_size,  # Specify your desired batch size here
        #     max_positions=max_positions_train,
        #     epoch=epoch_i,
        #     sample_without_replacement=args.sample_without_replacement,
        #     sort_by_source_size=(epoch_i <= args.curriculum),
        #     shard_id=args.distributed_rank,
        #     num_shards=args.distributed_world_size,
        # )


        # reset meters
        for key, val in g_logging_meters.items():
            if val is not None:
                val.reset()
        for key, val in d_logging_meters.items():
            if val is not None:
                val.reset()

        # set training mode
        generator.train()
        discriminator.train()
        update_learning_rate(num_update, 8e4, args.g_learning_rate, args.lr_shrink, g_optimizer)
        
        generator_pt.eval()
        
        for i, sample in tqdm(enumerate(trainloader)):
            
            print("type of sample: ", type(sample))
            print("keys of sample: ", sample.keys())
            print("dictionary sample size : ", len(sample))

            if use_cuda:
                # wrap input tensors in cuda tensors
                sample = utils.make_variable(sample, cuda=cuda)

            ############# GENERATOR ####################################
            # MLE training
            print("MLE Training")
            
            ##*****************************************************************************
            # Generate Translation with Fairseq Model
            src_tokens = sample['net_input']['src_tokens']
            # make sure that the tokens are on the right device
            src_tokens = src_tokens.cuda() if use_cuda else src_tokens
            # print("src_tokens type: ", type(src_tokens))
            # print("src_tokens size() : ", src_tokens.size())
            # print("src_tokens : ", src_tokens)
            # Generate translations using the Fairseq Model 
            
            # src_tokens to be converted to a En sentence ( EN-> FR), remove bpe 
            
            sentences = ids_to_sentences(src_tokens, dataset.src_dict)
            # print("sentences :::", sentences)
            # for sentence in sentences:
            #     # print("sentence no#  ", n)
            #     print(sentence)
                
            # Access the original TransformerModel from the DataParallel wrapper
            original_generator_pt = generator_pt.module if isinstance(generator_pt, torch.nn.DataParallel) else generator_pt

            # Now use the translate method
            translations = original_generator_pt.translate(sentences)

            #translations = generator_pt.translate(sentences)
            # Assuming that your generator model expects a certain format, you might need to convert translation to that format
            # print("translations ", translations)
            
            # Applying BPE to translations
            bpe_translations = [apply_bpe(t) for t in translations]
            # print("translations with BPE:", bpe_translations)
            
            # Applying padding to bpe_translations 
            #  max_len to be 50 for demonstration purposes
            max_len = 50
            padded_bpe_translations = pad_sentences(bpe_translations, max_len=max_len)
            # print("Padded translations with BPE:", padded_bpe_translations)
            
            # convert bpe_translations to ids 
            # sys_out_batch = convert_to_expected_format(translation)
            
            
            max_len = 50  # Example maximum length
            #token_ids_tensor = sentences_to_ids(padded_bpe_translations, dataset.dst_dict, max_len) # The dataset.dst_dict ( here FR ) should be from data-bin/, which the GAN train() is used
            #token_ids_tensor = sentences_to_ids(padded_bpe_translations, dataset.dst_dict, max_len) # The dataset.dst_dict ( here FR ) should be from data-bin/, which the GAN train() is used
            dict_obj = Dictionary()
            token_ids_tensor = dict_obj.sentences_to_ids(padded_bpe_translations, max_len=50)
            
            # Check shape of the resulting tensor
            # print("token_ids_tensor.shape ", token_ids_tensor.shape)
            token_ids_tensor_flat = token_ids_tensor.view(-1)
            # print("token_ids_tensor_flat.shape ", token_ids_tensor_flat.shape)
            
            ##*****************************************************************************
            
            sys_out_batch = generator(sample=sample, args=args)
            print("sys_out_batch ", sys_out_batch)
            print("sys_out_batch size: ", sys_out_batch.size())

            out_batch = sys_out_batch.contiguous().view(-1, sys_out_batch.size(-1)) # (64 X 50) X 6632  
            print("out_batch size: ", out_batch.size())

            train_trg_batch = sample['target'].view(-1) # 64*50 = 3200

            loss = g_criterion(out_batch, train_trg_batch)

            sample_size = sample['target'].size(0) if args.sentence_avg else sample['ntokens']
            nsentences = sample['target'].size(0)
            logging_loss = loss.data / sample_size / math.log(2)
            g_logging_meters['bsz'].update(nsentences)
            g_logging_meters['train_loss'].update(logging_loss, sample_size)
            logging.debug(f"G MLE loss at batch {i}: {g_logging_meters['train_loss'].avg:.3f}, lr={g_optimizer.param_groups[0]['lr']}")
            g_optimizer.zero_grad()
            loss.backward()
            # all-reduce grads and rescale by grad_denom
            for p in generator.parameters():
                if p.requires_grad:
                    p.grad.data.div_(sample_size)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), args.clip_norm)
            g_optimizer.step()
            
            
            num_update += 1


            # part II: train the discriminator
            bsz = sample['target'].size(0) # batch_size = 64
        
            src_sentence = sample['net_input']['src_tokens'] # 64 x max-len i.e 64 X 50

            # now train with machine translation output i.e generator output
            true_sentence = sample['target'].view(-1) # 64*50 = 3200
            
            true_labels = Variable(torch.ones(sample['target'].size(0)).float()) # 64 length vector

            ###########################################################################################################
            
            with torch.no_grad():
                sys_out_batch = generator(sample=sample, args=args) # 64 X 50 X 6632
                
            out_batch = sys_out_batch.contiguous().view(-1, sys_out_batch.size(-1)) # (64 X 50) X 6632  
                
            _,prediction = out_batch.topk(1)
            # print("prediction before squeeze: ",prediction.size())
            prediction = prediction.squeeze(1)  #64 * 50 = 6632
            # print("prediction after squeeze: ",prediction.size())
            
            
            ##############################################################################################################################
            
            # prediction = token_ids_tensor_flat
            
            # print("prediction after token_ids_tensor_flat: ",prediction.size())
            
            if use_cuda:
                token_ids_tensor_flat = token_ids_tensor_flat.cuda()
            
            fake_labels = Variable(torch.zeros(sample['target'].size(0)).float()) # 64 length vector

            #print("fake_labels after Variable: ",fake_labels.size())
            
            fake_sentence = torch.reshape(prediction, src_sentence.shape) # 64 X 50 

            #print("fake_sentence after torch.reshape(prediction, src_sentence.shape) : ",fake_sentence.size())
            
            #############################################################################################################################
            
            fake_sentence_gpt = torch.reshape(token_ids_tensor_flat, src_sentence.shape)
            
            #print("fake_sentence_gpt after torch.resshape(token_ids_tensor_flat, src_sentence.shape) : ",fake_sentence_gpt.size())
            
            #############################################################################################################################
            
            if use_cuda:
                true_labels = true_labels.cuda()
            
            true_sentence = torch.reshape(true_sentence, src_sentence.shape) # 64 X 50 
            disc_out_humanTranSent = discriminator(src_sentence, true_sentence)
            d_loss_human = d_criterion(disc_out_humanTranSent.squeeze(1), true_labels)
            
            if use_cuda:
                fake_labels = fake_labels.cuda()
            
            disc_out = discriminator(src_sentence, fake_sentence) # 64 X 1
            
            d_loss_fakeG = d_criterion(disc_out.squeeze(1), fake_labels)
            
            ##################### pre-trained loss ########         
            if use_cuda:
                src_sentence = src_sentence.cuda()
                fake_labels = fake_labels.cuda()
            
            disc_out_gpt = discriminator(src_sentence, fake_sentence_gpt)
            
            d_loss_gpt = d_criterion(disc_out_gpt.squeeze(1), fake_labels)
            
            ######################################################################################################
            # d_loss = 0.5*(d_loss + d_loss_human)
            
            # d_loss = 0.20*d_loss_fakeG + 0.40*d_loss_gpt + 0.40*d_loss_human
            # d_loss = 0.50*d_loss_gpt + 0.50*d_loss_human
            # d_loss = 0.80*d_loss_gpt + 0.20*d_loss_human
            d_loss = d_loss_gpt

            acc = torch.sum(torch.round(disc_out).squeeze(1) == fake_labels).float() / len(fake_labels)

            d_logging_meters['train_acc'].update(acc)
            d_logging_meters['train_loss'].update(d_loss)
            logging.debug(f"D training loss {d_logging_meters['train_loss'].avg:.3f}, acc {d_logging_meters['train_acc'].avg:.3f} at batch {i}")
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()


        # saving training check-points of generator and discriminator 
        torch.save(generator,
                   open(checkpoints_path + f"train_joint_g_{g_logging_meters['train_loss'].avg:.3f}.epoch_{epoch_i}.pt",
                        'wb'), pickle_module=dill)
        print("saving discriminator at training: ")
        torch.save(discriminator,
                   open(checkpoints_path + f"train_joint_d_{d_logging_meters['train_loss'].avg:.3f}.epoch_{epoch_i}.pt",
                        'wb'), pickle_module=dill)
"""
 ################################################################## VALIDATION #################################################################
        # validation
        # set validation mode
        generator.eval()
        discriminator.eval()
        # Initialize dataloader
        max_positions_valid = (args.fixed_max_len, args.fixed_max_len)
        valloader = dataset.eval_dataloader(
            'valid',
            max_tokens=args.max_tokens,
            max_sentences=args.joint_batch_size,
            max_positions=max_positions_valid,
            skip_invalid_size_inputs_valid_test=True,
            descending=True,  # largest batch first to warm the caching allocator
            shard_id=args.distributed_rank,
            num_shards=args.distributed_world_size,
        )

        # reset meters
        for key, val in g_logging_meters.items():
            if val is not None:
                val.reset()
        for key, val in d_logging_meters.items():
            if val is not None:
                val.reset()


        for i, sample in tqdm(enumerate(valloader)):
            
            # print statements for debugging purposes
            print("####################### the value of i in valloader is {} #######################".format(i))

            with torch.no_grad():
                if use_cuda:
                    # wrap input tensors in cuda tensors
                    sample = utils.make_variable(sample, cuda=cuda)

                # generator validation
                sys_out_batch = generator(sample=sample, args=args)
                out_batch = sys_out_batch.contiguous().view(-1, sys_out_batch.size(-1)) # (64 X 50) X 6632  
                dev_trg_batch = sample['target'].view(-1) # 64*50 = 3200

                loss = g_criterion(out_batch, dev_trg_batch)
                sample_size = sample['target'].size(0) if args.sentence_avg else sample['ntokens']
                loss = loss / sample_size / math.log(2)
                g_logging_meters['valid_loss'].update(loss, sample_size)
                logging.debug(f"G dev loss at batch {i}: {g_logging_meters['valid_loss'].avg:.3f}")

                # discriminator validation
                bsz = sample['target'].size(0)
                # print("bsz value is {}".format(bsz))
                src_sentence = sample['net_input']['src_tokens']
                # train with half human-translation and half machine translation

                true_sentence = sample['target']
                true_labels = Variable(torch.ones(sample['target'].size(0)).float())

                
                # print statements for debugging purposes
                # print("sample type is {}".format(type(sample)))
                # print("sample keys are {}".format(sample.keys()))
                # print("sample dict length is {}".format(len(sample)))
                
                # print("sample['id] ", type(sample['id']))
                # print("sample['id] size ", sample['id'].size())
                
                # print("sample[''ntokens'] ", type(sample['ntokens']))
                # print("sample['ntokens'] ", sample['ntokens'])
                
                # print("sample[''net_input'] ", type(sample['net_input']))
                # print("sample['net_input'] ", sample['net_input'].keys())
                
                # print("sample['target'] ", type(sample['target']))
                # print("sample['target'] ", sample['target'].size())
                
                # print("sample['net_input']['src_tokens'] ", sample['net_input']['src_tokens'])
                # print("sample['net_input']['prev_output_tokens'] ", sample['net_input']['prev_output_tokens'])
            
                with torch.no_grad():
                    sys_out_batch = generator(sample=sample, args=args)
                
                # print statements for debugging purposes
                # print("sys_out_batch size is {}".format(sys_out_batch.size()))

                out_batch = sys_out_batch.contiguous().view(-1, sys_out_batch.size(-1)) # (64 X 50) X 6632  
                
                # print statements for debugging purposes
                # print("out_batch size is {}".format(out_batch.size()))

                _,prediction = out_batch.topk(1)
                
                # print statements for debugging purposes
                # print("prediction size is {}".format(prediction.size()))
                
                prediction = prediction.squeeze(1)  #64 * 50 = 6632
                
                # print statements for debugging purposes
                # print("prediction.squeeze(1) value  is {}".format(prediction))
                # print("prediction.squeeze(1) type  is {}".format(type(prediction)))

                fake_labels = Variable(torch.zeros(sample['target'].size(0)).float())

                fake_sentence = torch.reshape(prediction, src_sentence.shape) # 64 X 50 

                if use_cuda:
                    fake_labels = fake_labels.cuda()
                    
                # print("This {} is the source sentence type".format(type(src_sentence)))
                # print("This {} is the source sentence size".format(src_sentence.size()))
                # print("This {} is the fake sentence(generated by G)type".format(type(fake_sentence)))
                # print("This {} is the fake sentence(generated by G)size".format(fake_sentence.size()))
                
                
                disc_out = discriminator(src_sentence, fake_sentence)
                
                # print("This {} is the disc_out type".format(type(disc_out)))
                # print("This {} is the disc_out size".format(disc_out.size()))
                # print("This {} is the disc_out.squeeze(1) type".format(type(disc_out.squeeze(1))))
                # print("This {} is the disc_out.squeeze(1) size".format(disc_out.squeeze(1).size()))
                # print("This {} is the disc_out.squeeze(1) value".format(disc_out.squeeze(1)))
                
                d_loss = d_criterion(disc_out.squeeze(1), fake_labels)
                
                if use_cuda:
                    true_labels = true_labels.cuda()
            
                true_sentence = torch.reshape(true_sentence, src_sentence.shape) # 64 X 50 
                
                disc_out_humanTranSent = discriminator(src_sentence, true_sentence)
                d_loss_human = d_criterion(disc_out_humanTranSent.squeeze(1), true_labels)
                d_loss = 0.7*d_loss + 0.3*d_loss_human

                acc = torch.sum(torch.round(disc_out).squeeze(1) == fake_labels).float() / len(fake_labels)
                d_logging_meters['valid_acc'].update(acc)
                d_logging_meters['valid_loss'].update(d_loss)
                logging.debug(f"D dev loss {d_logging_meters['valid_loss'].avg:.3f}, acc {d_logging_meters['valid_acc'].avg:.3f} at batch {i}")

        print("saving generator: ")
        torch.save(generator,
                   open(checkpoints_path + f"test_joint_g_{g_logging_meters['valid_loss'].avg:.3f}.epoch_{epoch_i}.pt",
                        'wb'), pickle_module=dill)
        print("saving discriminator: ")
        torch.save(discriminator,
                   open(checkpoints_path + f"test_joint_d_{d_logging_meters['valid_loss'].avg:.3f}.epoch_{epoch_i}.pt",
                        'wb'), pickle_module=dill)

        if g_logging_meters['valid_loss'].avg < best_dev_loss:
            best_dev_loss = g_logging_meters['valid_loss'].avg
            torch.save(generator, open(checkpoints_path + "tf_disc_best_gmodel.pt", 'wb'), pickle_module=dill)
            torch.save(discriminator, open(checkpoints_path + "tf_disc_best_dmodel_at_best_gmodel.pt", 'wb'), pickle_module=dill)
        
###############################################################################################################################################
"""
def update_learning_rate(update_times, target_times, init_lr, lr_shrink, optimizer):

    lr = init_lr * (lr_shrink ** (update_times // target_times))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == "__main__":
  ret = parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning(f"unknown arguments: {parser.parse_known_args()[1]}")
  main(options)