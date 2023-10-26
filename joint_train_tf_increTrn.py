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
sys.path.append("/u/prattisr/phase-2/all_repos/Adversarial_NMT/neural-machine-translation-using-gan-master")
# https://stackoverflow.com/questions/67311527/how-to-set-gpu-count-to-0-using-os-environcuda-visible-devices
"""
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
torch.cuda.device_count() # result is 2

os.environ["CUDA_VISIBLE_DEVICES"]="0"
torch.cuda.device_count() # result is 1, using first GPU

os.environ["CUDA_VISIBLE_DEVICES"]="1"
torch.cuda.device_count() # result is 1, using second GPU"""
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch import cuda
from torch.autograd import Variable

import data
import utils
from meters import AverageMeter
from discriminator import Discriminator
# from generator_tf import TransformerModel
from generator_tf_ptfseq import TransformerModel_custom
from train_generator_new import train_g
from train_discriminator import train_d
from PGLoss import PGLoss
from tqdm import tqdm



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
    use_cuda = (len(args.gpuid) >= 1)
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

    # loading n-1 iteration g model 
   # generator = TransformerModel(args, dataset.src_dict, dataset.dst_dict, use_cuda=use_cuda)
    g_model_path = '/u/prattisr/phase-2/all_repos/Adversarial_NMT/neural-machine-translation-using-gan-master/pretrained_models/checkpoints/joint/test_wmt14_en_fr_2023_pt_oc5_sm_50k_v1/test_joint_g_9.846.epoch_1.pt'
    assert os.path.exists(g_model_path)
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
    # generator.train()
    
    print("Generator n-1 pretrained TransformerModel loaded successfully!")
    
    
    ####################### Discriminator Loading ##################################
    # discriminator = Discriminator(args, dataset.src_dict, dataset.dst_dict, use_cuda=use_cuda)
    d_model_path = '/u/prattisr/phase-2/all_repos/Adversarial_NMT/neural-machine-translation-using-gan-master/pretrained_models/checkpoints/joint/test_wmt14_en_fr_2023_pt_oc5_sm_50k_v1/test_joint_d_0.428.epoch_1.pt'
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
    
    # # fix discriminator word embedding (as Wu et al. do)
    # for p in discriminator.embed_src_tokens.parameters():
    #     p.requires_grad = False
    # for p in discriminator.embed_trg_tokens.parameters():
    #     p.requires_grad = False
    
    print("Discriminator n-1 pretrained  loaded successfully!")

    
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

    # adversarial training checkpoints saving path
    if not os.path.exists('checkpoints/joint/test_wmt14_en_fr_2023_pt_oc5_sm_50k_v1'):
        os.makedirs('checkpoints/joint/test_wmt14_en_fr_2023_pt_oc5_sm_50k_v1')
    checkpoints_path = 'checkpoints/joint/test_wmt14_en_fr_2023_pt_oc5_sm_50k_v1/'

    # define loss function
    g_criterion = torch.nn.NLLLoss(ignore_index=dataset.dst_dict.pad(),reduction='sum')
    d_criterion = torch.nn.BCELoss()
    pg_criterion = PGLoss(ignore_index=dataset.dst_dict.pad(), size_average=True,reduce=True)

    # fix discriminator word embedding (as Wu et al. do)
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

    # start joint training
    best_dev_loss = math.inf
    num_update = 0
    # main training loop
    for epoch_i in tqdm(range(1, args.epochs + 1)):
        logging.info("At {0}-th epoch.".format(epoch_i))

        seed = args.seed + epoch_i
        torch.manual_seed(seed)

        max_positions_train = (args.fixed_max_len, args.fixed_max_len)

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

        for i, sample in tqdm(enumerate(trainloader)):

            if use_cuda:
                # wrap input tensors in cuda tensors
                sample = utils.make_variable(sample, cuda=cuda)

            ## part I: use gradient policy method to train the generator

            # use policy gradient training when random.random() > 50%
            # if random.random()  >= 0.5:
            if 0.2  >= 0.5:

                print("Policy Gradient Training")
                
                # sys_out_batch = generator(sample) # 64 X 50 X 6632
                sys_out_batch = generator(sample=sample, args=args)

                out_batch = sys_out_batch.contiguous().view(-1, sys_out_batch.size(-1)) # (64 * 50) X 6632   
                 
                _,prediction = out_batch.topk(1)
                prediction = prediction.squeeze(1) # 64*50 = 3200
                prediction = torch.reshape(prediction, sample['net_input']['src_tokens'].shape) # 64 X 50
                
                with torch.no_grad():
                    reward = discriminator(sample['net_input']['src_tokens'], prediction) # 64 X 1

                train_trg_batch = sample['target'] # 64 x 50
                
                pg_loss = pg_criterion(sys_out_batch, train_trg_batch, reward, use_cuda)
                sample_size = sample['target'].size(0) if args.sentence_avg else sample['ntokens'] # 64
                logging_loss = pg_loss / math.log(2)
                g_logging_meters['train_loss'].update(logging_loss.item(), sample_size)
                logging.debug(f"G policy gradient loss at batch {i}: {pg_loss.item():.3f}, lr={g_optimizer.param_groups[0]['lr']}")
                g_optimizer.zero_grad()
                pg_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), args.clip_norm)
                g_optimizer.step()

            else:
                # MLE training
                print("MLE Training")

                sys_out_batch = generator(sample=sample, args=args)

                out_batch = sys_out_batch.contiguous().view(-1, sys_out_batch.size(-1)) # (64 X 50) X 6632  

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

            with torch.no_grad():
                sys_out_batch = generator(sample=sample, args=args) # 64 X 50 X 6632
                
            out_batch = sys_out_batch.contiguous().view(-1, sys_out_batch.size(-1)) # (64 X 50) X 6632  
                
            _,prediction = out_batch.topk(1)
            prediction = prediction.squeeze(1)  #64 * 50 = 6632
            
            fake_labels = Variable(torch.zeros(sample['target'].size(0)).float()) # 64 length vector

            fake_sentence = torch.reshape(prediction, src_sentence.shape) # 64 X 50 

            
            # To-Do : Can add ? ? Need to research a bit more on this 
            # disc_out_humanTranSent = discriminator(src_sentence, true_sentence)
            # d_loss_human = d_criterion(disc_out_humanTranSent.squeeze(1), true_labels)
            # then d_loss = Average(d_loss + d_loss_human)
            
            if use_cuda:
                true_labels = true_labels.cuda()
            
            true_sentence = torch.reshape(true_sentence, src_sentence.shape) # 64 X 50 
            disc_out_humanTranSent = discriminator(src_sentence, true_sentence)
            d_loss_human = d_criterion(disc_out_humanTranSent.squeeze(1), true_labels)
            # then d_loss = Average(d_loss + d_loss_human)
            
            if use_cuda:
                fake_labels = fake_labels.cuda()
            
            disc_out = discriminator(src_sentence, fake_sentence) # 64 X 1
            
            d_loss = d_criterion(disc_out.squeeze(1), fake_labels)
            
            # 
            # d_loss = 0.5*(d_loss + d_loss_human)
            d_loss = 0.7*d_loss + 0.3*d_loss_human

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
                print("bsz value is {}".format(bsz))
                src_sentence = sample['net_input']['src_tokens']
                # train with half human-translation and half machine translation

                true_sentence = sample['target']
                true_labels = Variable(torch.ones(sample['target'].size(0)).float())

                
                # print statements for debugging purposes
                print("sample type is {}".format(type(sample)))
                print("sample keys are {}".format(sample.keys()))
                print("sample dict length is {}".format(len(sample)))
                
                print("sample['id] ", type(sample['id']))
                print("sample['id] size ", sample['id'].size())
                
                print("sample[''ntokens'] ", type(sample['ntokens']))
                print("sample['ntokens'] ", sample['ntokens'])
                
                print("sample[''net_input'] ", type(sample['net_input']))
                print("sample['net_input'] ", sample['net_input'].keys())
                
                print("sample['target'] ", type(sample['target']))
                print("sample['target'] ", sample['target'].size())
                
                print("sample['net_input']['src_tokens'] ", sample['net_input']['src_tokens'])
                print("sample['net_input']['prev_output_tokens'] ", sample['net_input']['prev_output_tokens'])
            
                with torch.no_grad():
                    sys_out_batch = generator(sample=sample, args=args)
                
                # print statements for debugging purposes
                print("sys_out_batch size is {}".format(sys_out_batch.size()))

                out_batch = sys_out_batch.contiguous().view(-1, sys_out_batch.size(-1)) # (64 X 50) X 6632  
                
                # print statements for debugging purposes
                print("out_batch size is {}".format(out_batch.size()))

                _,prediction = out_batch.topk(1)
                
                # print statements for debugging purposes
                print("prediction size is {}".format(prediction.size()))
                
                prediction = prediction.squeeze(1)  #64 * 50 = 6632
                
                # print statements for debugging purposes
                print("prediction.squeeze(1) value  is {}".format(prediction))
                print("prediction.squeeze(1) type  is {}".format(type(prediction)))

                fake_labels = Variable(torch.zeros(sample['target'].size(0)).float())

                fake_sentence = torch.reshape(prediction, src_sentence.shape) # 64 X 50 

                if use_cuda:
                    fake_labels = fake_labels.cuda()
                    
                print("This {} is the source sentence type".format(type(src_sentence)))
                print("This {} is the source sentence size".format(src_sentence.size()))
                print("This {} is the fake sentence(generated by G)type".format(type(fake_sentence)))
                print("This {} is the fake sentence(generated by G)size".format(fake_sentence.size()))
                
                """
                This <class 'torch.Tensor'> is the source sentence type
                This torch.Size([1, 50]) is the source sentence size
                This <class 'torch.Tensor'> is the fake sentence(generated by G)type
                This torch.Size([1, 50]) is the fake sentence(generated by G)size
                """
                
                disc_out = discriminator(src_sentence, fake_sentence)
                
                print("This {} is the disc_out type".format(type(disc_out)))
                print("This {} is the disc_out size".format(disc_out.size()))
                print("This {} is the disc_out.squeeze(1) type".format(type(disc_out.squeeze(1))))
                print("This {} is the disc_out.squeeze(1) size".format(disc_out.squeeze(1).size()))
                print("This {} is the disc_out.squeeze(1) value".format(disc_out.squeeze(1)))
                
                """
                This <class 'torch.Tensor'> is the disc_out type
                This torch.Size([1, 1]) is the disc_out size
                This <class 'torch.Tensor'> is the disc_out.squeeze(1) type
                This torch.Size([1]) is the disc_out.squeeze(1) size
                This tensor([0.0113], device='cuda:0') is the disc_out.squeeze(1) value
                """
                d_loss = d_criterion(disc_out.squeeze(1), fake_labels)
                
                if use_cuda:
                    true_labels = true_labels.cuda()
            
                true_sentence = torch.reshape(true_sentence, src_sentence.shape) # 64 X 50 
                
                disc_out_humanTranSent = discriminator(src_sentence, true_sentence)
                d_loss_human = d_criterion(disc_out_humanTranSent.squeeze(1), true_labels)
                d_loss = 0.5*(d_loss + d_loss_human)

                acc = torch.sum(torch.round(disc_out).squeeze(1) == fake_labels).float() / len(fake_labels)
                d_logging_meters['valid_acc'].update(acc)
                d_logging_meters['valid_loss'].update(d_loss)
                logging.debug(f"D dev loss {d_logging_meters['valid_loss'].avg:.3f}, acc {d_logging_meters['valid_acc'].avg:.3f} at batch {i}")

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
        
        # if g_logging_meters['valid_loss'].avg < best_dev_loss:
        #     best_dev_loss = g_logging_meters['valid_loss'].avg
        #     torch.save(generator, open(checkpoints_path + "best_gmodel.pt", 'wb'), pickle_module=dill)


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