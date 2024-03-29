'''

This code is adapted from Facebook Fairseq-py
Visit https://github.com/facebookresearch/fairseq-py for more information

'''
import os
import sys
getpwd = os.getcwd()
# sys.path.append(
#     "/u/prattisr/phase-2/all_repos/Adversarial_NMT/neural-machine-translation-using-gan-master"
# )
sys.path.append(getpwd)
import contextlib
import itertools
import glob
import math
import numbers
import numpy as np
import os
import torch
import torch.utils.data
from dictionary import Dictionary
from indexed_dataset import IndexedDataset, IndexedInMemoryDataset, IndexedRawTextDataset, IndexedRawTextDataset_label


def has_binary_files(data_dir, splits):
    for split in splits:
        if len(glob.glob(os.path.join(data_dir, '{}.*-*.*.bin'.format(split)))) < 2:
            return False
    return True


def infer_language_pair(path, splits):
    """Infer language pair from filename: <split>.<lang1>-<lang2>.(...).idx"""
    src, dst = None, None
    for filename in os.listdir(path):
        parts = filename.split('.')
        for split in splits:
            if parts[0] == split and parts[-1] == 'idx':
                src, dst = parts[1].split('-')
                break
    return src, dst


def load_dictionaries(path, src_lang, dst_lang):
    """Load dictionaries for a given language pair."""
    src_dict = Dictionary.load(os.path.join(path, 'dict.{}.txt'.format(src_lang)))
    dst_dict = Dictionary.load(os.path.join(path, 'dict.{}.txt'.format(dst_lang)))    
    return src_dict, dst_dict

from torch._utils import _accumulate
from torch import randperm



def load_dataset(path, load_splits, src=None, dst=None, maxlen=None):
    """Loads specified data splits (e.g., test, train or valid) from the
    specified folder and check that files exist."""
    if src is None and dst is None:
        # find language pair automatically
        src, dst = infer_language_pair(path, load_splits)
    assert src is not None and dst is not None, 'Source and target languages should be provided'

    src_dict, dst_dict = load_dictionaries(path, src, dst)
    
    dataset = LanguageDatasets(src, dst, src_dict, dst_dict)
    
    # Load dataset from binary files
    def all_splits_exist(src, dst):
        for split in load_splits:
            filename = '{0}.{1}-{2}.{1}.idx'.format(split, src, dst)
            if not os.path.exists(os.path.join(path, filename)):
                return False
        return True
        
    # infer langcode
    if all_splits_exist(src, dst):
        langcode = '{}-{}'.format(src, dst)
    elif all_splits_exist(dst, src):
        langcode = '{}-{}'.format(dst, src)
    else:
        raise Exception('Dataset cannot be loaded from path: ' + path)

    def fmt_path(fmt, *args):
        return os.path.join(path, fmt.format(*args))

    for split in load_splits:
        prefix = split
        src_path = fmt_path('{}.{}.{}', prefix, langcode, src)
        dst_path = fmt_path('{}.{}.{}', prefix, langcode, dst)

        if not IndexedInMemoryDataset.exists(src_path):
            break

        dataset.splits[prefix] = LanguagePairDataset(
            IndexedInMemoryDataset(src_path),
            IndexedInMemoryDataset(dst_path),
            pad_idx=dataset.src_dict.pad(),
            eos_idx=dataset.src_dict.eos(),
            maxlen=maxlen
        )
    # lengths = [50000, len(dataset.splits['train'])-50000]

    # dataset.splits['train']  = random_split(dataset.splits['train'],lengths)[0]
    # print(type(dataset.splits['train']))


    return dataset


def load_raw_text_dataset(path, load_splits, src=None, dst=None, maxlen=None):
    """Loads specified data splits (e.g., test, train or valid) from raw text
    files in the specified folder."""
    # print("{} is the source \n".format(src))
    # print("{} is the destination \n".format(dst))
    # print("{} is the maxlen \n".format(maxlen))
    
    if src is None and dst is None:
        # find language pair automatically
        src, dst = infer_language_pair(path, load_splits)
    assert src is not None and dst is not None, 'Source and target languages should be provided'

    print("path: \n", path)
    src_dict, dst_dict = load_dictionaries(path, src, dst)
    print("{} : is source dictionary".format(len(src_dict)))
    print("{} : is destination dictionary".format(len(dst_dict)))
    dataset = LanguageDatasets(src, dst, src_dict, dst_dict)

    # Load dataset from raw text files
    for split in load_splits:
        print("split ",split)
        src_path = os.path.join(path, '{}.{}'.format(split, src))
        dst_path = os.path.join(path, '{}.{}'.format(split, dst))
        print("src_path : ", src_path)
        print("src_dict : ", src_dict)
        print("dst_path : ", dst_path)
        print("dst_dict : ", dst_dict)
        dataset.splits[split] = LanguagePairDataset(
            IndexedRawTextDataset(src_path, src_dict),
            IndexedRawTextDataset(dst_path, dst_dict),
            pad_idx=dataset.src_dict.pad(),
            eos_idx=dataset.src_dict.eos(),
            maxlen=maxlen
        )
    print("dataset: ", dataset)    
    return dataset

def load_raw_text_dataset_test_classify(path, load_splits, src=None, dst=None, maxlen=None):
    """Loads specified data splits (e.g., test, train or valid) from raw text
    files in the specified folder."""

    # print("{} is the source in load_raw_text_dataset_test_classify\n".format(src))
    # print("{} is the destination in load_raw_text_dataset_test_classify\n".format(dst))
    # print("{} is the maxlen in load_raw_text_dataset_test_classify\n".format(maxlen))

    assert src is not None and dst is not None, 'Source and target languages should be provided'

    print("path: \n", path)
    src_dict, dst_dict = load_dictionaries(path, src, dst)

    # print("{} : is source dictionary load_raw_text_dataset_test_classify".format(len(src_dict)))
    # print("{} : is destination dictionary load_raw_text_dataset_test_classify".format(len(dst_dict)))
# to-do : Dec 5th - 2022 
    dataset = LanguageDatasets_test_classify(src, dst, src_dict, dst_dict)

    # Load dataset from raw text files
    for split in load_splits:
        print("split ",split)

        src_path = os.path.join(path, 'src.en')
        dst_path = os.path.join(path, 'target.fr')
        ht_mt_path = os.path.join(path, 'ht_mt_target.fr')
        ht_mt_label_path = os.path.join(path, 'ht_mt_label')

        # print("src_path : ", src_path)
        # print("src_dict : ", src_dict)
        # print("dst_path : ", dst_path)
        # print("dst_dict : ", dst_dict)
        
        # print("ht_mt_path ", ht_mt_path)
        # print("ht_mt_label_path ", ht_mt_label_path)
        
        dataset.splits[split] = LanguagePairDataset_test_classify(
            IndexedRawTextDataset(src_path, src_dict),
            IndexedRawTextDataset(dst_path, dst_dict),
            IndexedRawTextDataset(ht_mt_path, dst_dict),
            IndexedRawTextDataset_label(ht_mt_label_path),
            pad_idx=dataset.src_dict.pad(),
            eos_idx=dataset.src_dict.eos(),
            maxlen=maxlen)
        # )
    print("dataset: ", dataset)    
    return dataset

class LanguageDatasets(object):
    def __init__(self, src, dst, src_dict, dst_dict):
        self.src = src
        self.dst = dst
        self.src_dict = src_dict
        self.dst_dict = dst_dict
        self.splits = {}

        assert self.src_dict.pad() == self.dst_dict.pad()
        assert self.src_dict.eos() == self.dst_dict.eos()
        assert self.src_dict.unk() == self.dst_dict.unk()

    def train_dataloader(self, split, max_tokens=None,
                         max_sentences=None, max_positions=(1024, 1024),
                         seed=None, epoch=1, sample_without_replacement=0,
                         sort_by_source_size=False, shard_id=0, num_shards=1):
        dataset = self.splits[split]
        print("dataset in train dataloader", dataset)
        with numpy_seed(seed):
            batch_sampler = shuffled_batches_by_size(
                dataset.src, dataset.dst, max_tokens=max_tokens,
                max_sentences=max_sentences, epoch=epoch,
                sample=sample_without_replacement, max_positions=max_positions,
                sort_by_source_size=sort_by_source_size)
            # Drop the last batch if it's smaller than the expected size
            if max_sentences and len(batch_sampler[-1]) < max_sentences:
                batch_sampler = batch_sampler[:-1]
            # batch_sampler = mask_batches(batch_sampler, shard_id=shard_id, num_shards=num_shards) 
            # When setting up your DataLoader
            batch_sampler = mask_batches(
                batch_sampler=batch_sampler, 
                shard_id=shard_id, 
                num_shards=num_shards, 
                drop_last=True  # Ensure the last incomplete batch is dropped
            )

        return torch.utils.data.DataLoader(
            dataset, collate_fn=dataset.collater,
            batch_sampler=batch_sampler)

    # Customized train_dataloader
    # from torch.utils.data import BatchSampler, SequentialSampler

    # def train_dataloader(self, split, batch_size, max_positions=(1024, 1024), seed=None, epoch=1, 
    #                     sample_without_replacement=0, sort_by_source_size=False, shard_id=0, num_shards=1):
    #     dataset = self.splits[split]
        
    #     # Create a sequential sampler
    #     sequential_sampler = torch.utils.data.SequentialSampler(dataset)
        
    #     # Create a batch sampler that extracts batches of fixed size from the sequential sampler
    #     fixed_size_batch_sampler = torch.utils.data.BatchSampler(sequential_sampler, batch_size=batch_size, drop_last=True)
        
    #     # You don't need to modify the batches for different GPUs as each will independently
    #     # iterate over the dataset using this fixed-size batch sampler.
    #     return torch.utils.data.DataLoader(dataset, 
    #                                     collate_fn=dataset.collater, batch_sampler=fixed_size_batch_sampler)



    def eval_dataloader(self, split, num_workers=0, max_tokens=None,
                        max_sentences=None, max_positions=(1024, 1024),
                        skip_invalid_size_inputs_valid_test=False,
                        descending=False, shard_id=0, num_shards=1):
        dataset = self.splits[split]
        batch_sampler = batches_by_size(
            dataset.src, dataset.dst, max_tokens, max_sentences,
            max_positions=max_positions,
            ignore_invalid_inputs=skip_invalid_size_inputs_valid_test,
            descending=descending)
        # Drop the last batch if it's smaller than the expected size
        if max_sentences and len(batch_sampler[-1]) < max_sentences:
            batch_sampler = batch_sampler[:-1]
        batch_sampler = mask_batches(batch_sampler, shard_id=shard_id, num_shards=num_shards)
        return torch.utils.data.DataLoader(
            dataset, num_workers=num_workers, collate_fn=dataset.collater,
            batch_sampler=batch_sampler)
# to-do : Dec 5th - 2022 
class LanguageDatasets_test_classify(object):
    def __init__(self, src, dst, src_dict, dst_dict):
        self.src = src
        self.dst = dst
        self.src_dict = src_dict
        self.dst_dict = dst_dict
        self.splits = {}

        assert self.src_dict.pad() == self.dst_dict.pad()
        assert self.src_dict.eos() == self.dst_dict.eos()
        assert self.src_dict.unk() == self.dst_dict.unk()

    # def train_dataloader(self, split, max_tokens=None,
    #                      max_sentences=None, max_positions=(1024, 1024),
    #                      seed=None, epoch=1, sample_without_replacement=0,
    #                      sort_by_source_size=False, shard_id=0, num_shards=1):
    #     dataset = self.splits[split]
    #     with numpy_seed(seed):
    #         batch_sampler = shuffled_batches_by_size(
    #             dataset.src, dataset.dst, max_tokens=max_tokens,
    #             max_sentences=max_sentences, epoch=epoch,
    #             sample=sample_without_replacement, max_positions=max_positions,
    #             sort_by_source_size=sort_by_source_size)
    #         batch_sampler = mask_batches(batch_sampler, shard_id=shard_id, num_shards=num_shards)
    #     return torch.utils.data.DataLoader(
    #         dataset, collate_fn=dataset.collater,
    #         batch_sampler=batch_sampler)
        
    def eval_dataloader_test_classify(self, split, num_workers=0, max_tokens=None,
                        max_sentences=None, max_positions=(1024, 1024),
                        skip_invalid_size_inputs_valid_test=True,
                        descending=False, shard_id=0, num_shards=1):
        dataset = self.splits[split]
        print("dataset: ", type(dataset))
        batch_sampler = batches_by_size_test_classify(
            dataset.src, dataset.dst, max_tokens, max_sentences,
            max_positions=max_positions,
            ignore_invalid_inputs=skip_invalid_size_inputs_valid_test,
            descending=descending)
        
        print("type of batch_sampler ",type(batch_sampler))
        print("len of batch sampler:", len(batch_sampler))
        # # Drop the last batch if it's smaller than the expected size
        # if max_sentences and len(batch_sampler[-1]) < max_sentences:
        #     batch_sampler = batch_sampler[:-1]
        # batch_sampler = mask_batches(batch_sampler, shard_id=shard_id, num_shards=num_shards)
        
        batch_sampler = mask_batches_test_classify(batch_sampler, shard_id=shard_id, num_shards=num_shards)
        return torch.utils.data.DataLoader(
            dataset, num_workers=num_workers, collate_fn=dataset.collater,
            batch_sampler=batch_sampler)
    
    def eval_dataloader(self, split, num_workers=0, max_tokens=None,
                    max_sentences=None, max_positions=(1024, 1024),
                    skip_invalid_size_inputs_valid_test=False,
                    descending=False, shard_id=0, num_shards=1):
        dataset = self.splits[split]
        print("dataset in eval_dataloader : ", type(dataset))
        batch_sampler = batches_by_size(
            dataset.src, dataset.dst, max_tokens, max_sentences,
            max_positions=max_positions,
            ignore_invalid_inputs=skip_invalid_size_inputs_valid_test,
            descending=descending)
        batch_sampler = mask_batches(batch_sampler, shard_id=shard_id, num_shards=num_shards)
        return torch.utils.data.DataLoader(
            dataset, num_workers=num_workers, collate_fn=dataset.collater,
            batch_sampler=batch_sampler)


class sharded_iterator(object):

    def __init__(self, itr, num_shards, shard_id):
        assert shard_id >= 0 and shard_id < num_shards
        self.itr = itr
        self.num_shards = num_shards
        self.shard_id = shard_id

    def __len__(self):
        return len(self.itr)

    def __iter__(self):
        for i, v in enumerate(self.itr):
            if i % self.num_shards == self.shard_id:
                yield v


class LanguagePairDataset(torch.utils.data.Dataset):

    # padding constants
    LEFT_PAD_SOURCE = False
    LEFT_PAD_TARGET = False

    def __init__(self, src, dst, pad_idx, eos_idx, maxlen=None):
        self.src = src
        self.dst = dst
        self.pad_idx = pad_idx
        self.eos_idx = eos_idx
        self.maxlen = maxlen

    def __getitem__(self, i):
        # subtract 1 for 0-based indexing
        source = self.src[i].long() - 1
        target = self.dst[i].long() - 1
        return {
            'id': i,
            'source': source,
            'target': target,
        }

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        return LanguagePairDataset.collate(samples, self.pad_idx, self.eos_idx, self.maxlen)

    @staticmethod
    def collate(samples, pad_idx, eos_idx, maxlen):
        if len(samples) == 0:
            return {}
        def merge(key, left_pad, move_eos_to_beginning=False):
            return LanguagePairDataset.collate_tokens(
                [s[key] for s in samples],
                pad_idx, eos_idx, left_pad, move_eos_to_beginning, maxlen
            )

        id = torch.LongTensor([s['id'] for s in samples])
        src_tokens = merge('source', left_pad=LanguagePairDataset.LEFT_PAD_SOURCE)
        target = merge('target', left_pad=LanguagePairDataset.LEFT_PAD_TARGET)
        # we create a shifted version of targets for feeding the
        # previous output token(s) into the next decoder step
        prev_output_tokens = merge(
            'target',
            left_pad=LanguagePairDataset.LEFT_PAD_TARGET,
            move_eos_to_beginning=True,
        )

        # sort by descending source length
        src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
        src_lengths, sort_order = src_lengths.sort(descending=True)
        id = id.index_select(0, sort_order)
        src_tokens = src_tokens.index_select(0, sort_order)
        prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
        target = target.index_select(0, sort_order)

        return {
            'id': id,
            'ntokens': sum(len(s['target']) for s in samples),
            'net_input': {
                'src_tokens': src_tokens,
                'src_lengths': src_lengths,
                'prev_output_tokens': prev_output_tokens,
            },
            'target': target,
        }

    @staticmethod
    def collate_tokens(values, pad_idx, eos_idx, left_pad, move_eos_to_beginning=False, maxlen=None):        
        if maxlen is not None:
            if not max([v.size(0) for v in values]) <= maxlen:
                maxlen = max([v.size(0) for v in values])
        size = max([v.size(0) for v in values]) if maxlen is None else maxlen
        res = values[0].new(len(values), size).fill_(pad_idx)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            if move_eos_to_beginning:
                assert src[-1] == eos_idx
                dst[0] = eos_idx
                dst[1:] = src[:-1]
            else:
                dst.copy_(src)

        for i, v in enumerate(values):
            if left_pad:
                copy_tensor(v, res[i][size-len(v):])
            else:
                copy_tensor(v, res[i][:len(v)])
        return res

class LanguagePairDataset_test_classify(torch.utils.data.Dataset):
    
    # padding constants
    LEFT_PAD_SOURCE = False
    LEFT_PAD_TARGET = False

    def __init__(self, src, dst, ht_mt, ht_mt_label, pad_idx, eos_idx, maxlen=None):
        self.src = src
        self.dst = dst
        self.ht_mt = ht_mt
        self.ht_mt_label = ht_mt_label
        self.pad_idx = pad_idx
        self.eos_idx = eos_idx
        self.maxlen = maxlen

    def __getitem__(self, i):
        # subtract 1 for 0-based indexing
        
        # print("self.src[i] ", self.src[i])
        # print("self.src[i].long() ", self.src[i].long())
        # print("self.src[i].long()-1 ", self.src[i].long()-1)
        
        source = self.src[i].long() - 1
        
        # print("source in init --getitem-- ", source)
        # print("i value ", i)
        # print("self.ht_mt_label[i] ", self.ht_mt_label[i])
        
        target = self.dst[i].long() - 1
        ht_mt_target = self.ht_mt[i].long() - 1
        ht_mt_label =self.ht_mt_label[i]
        
        return {
            'id': i,
            'source': source,
            'target': target,
            'ht_mt_target': ht_mt_target,
            'ht_mt_label': ht_mt_label, 
        }

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        return LanguagePairDataset_test_classify.collate(samples, self.pad_idx, self.eos_idx, self.maxlen)

    @staticmethod
    def collate(samples, pad_idx, eos_idx, maxlen):
        if len(samples) == 0:
            return {}
        def merge(key, left_pad, move_eos_to_beginning=False):
            return LanguagePairDataset_test_classify.collate_tokens(
                [s[key] for s in samples],
                pad_idx, eos_idx, left_pad, move_eos_to_beginning, maxlen
            )

        id = torch.LongTensor([s['id'] for s in samples])
        # print("id ", id.size())
        for s in samples:
            print("s type ", type(s))
            # print("s keys() ", s.keys())
            # print("s s['id'] ", s['id'])
            # print("s s['source'] ", s['source'])
            # print("s s['target'] ", s['target'])
            # print("s s['ht_mt_target'] ", s['ht_mt_target'])
            # print("s s['ht_mt_label'] ", s['ht_mt_label'])
        src_tokens = merge('source', left_pad=LanguagePairDataset_test_classify.LEFT_PAD_SOURCE)
        target = merge('target', left_pad=LanguagePairDataset_test_classify.LEFT_PAD_TARGET)
        ht_mt_target = merge('ht_mt_target', left_pad=LanguagePairDataset_test_classify.LEFT_PAD_TARGET)
        ht_mt_label = torch.LongTensor([s['ht_mt_label'] for s in samples])
        
        # we create a shifted version of targets for feeding the
        # previous output token(s) into the next decoder step
        prev_output_tokens = merge(
            'target',
            left_pad=LanguagePairDataset_test_classify.LEFT_PAD_TARGET,
            move_eos_to_beginning=True,
        )
        
        prev_output_tokens_ht_mt_target = merge(
            'ht_mt_target',
            left_pad=LanguagePairDataset_test_classify.LEFT_PAD_TARGET,
            move_eos_to_beginning=True,
        )

        # sort by descending source length
        src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
        src_lengths, sort_order = src_lengths.sort(descending=True)
        print("sort_order :", sort_order)
        id = id.index_select(0, sort_order)
        src_tokens = src_tokens.index_select(0, sort_order)
        prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
        target = target.index_select(0, sort_order)
        
        prev_output_tokens_ht_mt_target = prev_output_tokens_ht_mt_target.index_select(0, sort_order)
        ht_mt_target = ht_mt_target.index_select(0, sort_order)
        ht_mt_label = ht_mt_label.index_select(0, sort_order)
        
        return {
            'id': id,
            'ntokens': sum(len(s['target']) for s in samples),
            'net_input': {
                'src_tokens': src_tokens,
                'src_lengths': src_lengths,
                'prev_output_tokens': prev_output_tokens,
            },
            'target': target,
            'ht_mt_target_trans' :{
                'ht_mt_target' : ht_mt_target, # ht_mt_target
                'ht_mt_label' : ht_mt_label,
                'prev_output_tokens_ht_mt_target': prev_output_tokens_ht_mt_target,
            },
        }

    @staticmethod
    def collate_tokens(values, pad_idx, eos_idx, left_pad, move_eos_to_beginning=False, maxlen=None):        
        if maxlen is not None:
            if not max([v.size(0) for v in values]) <= maxlen:
                maxlen = max([v.size(0) for v in values])
        size = max([v.size(0) for v in values]) if maxlen is None else maxlen
        res = values[0].new(len(values), size).fill_(pad_idx)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            if move_eos_to_beginning:
                assert src[-1] == eos_idx
                dst[0] = eos_idx
                dst[1:] = src[:-1]
            else:
                dst.copy_(src)

        for i, v in enumerate(values):
            if left_pad:
                copy_tensor(v, res[i][size-len(v):])
            else:
                copy_tensor(v, res[i][:len(v)])
        # print("res type: ", type(res))
        return res

class Subset(LanguagePairDataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

def random_split(dataset, lengths):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths
    ds
    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (iterable): lengths of splits to be produced
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths))
    return [Subset(dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]

def _valid_size(src_size, dst_size, max_positions):
    print("max_positions ",max_positions)
    if isinstance(max_positions, numbers.Number):
        max_src_positions, max_dst_positions = max_positions, max_positions
    else:
        max_src_positions, max_dst_positions = max_positions
    if src_size < 2 or src_size > max_src_positions:
        return False
    if dst_size is not None and (dst_size < 2 or dst_size > max_dst_positions):
        return False
    return True


def _make_batches(src, dst, indices, max_tokens, max_sentences, max_positions,
                  ignore_invalid_inputs=False, allow_different_src_lens=False):
    batch = []

    def yield_batch(next_idx, num_tokens):
        if len(batch) == 0:
            return False
        if len(batch) == max_sentences:
            return True
        if num_tokens > max_tokens:
            return True
        if not allow_different_src_lens and \
                (src.sizes[batch[0]] != src.sizes[next_idx]):
            return True
        return False

    sample_len = 0
    ignored = []
    for idx in map(int, indices):
        if not _valid_size(src.sizes[idx], dst.sizes[idx], max_positions):
            if ignore_invalid_inputs:
                ignored.append(idx)
                continue
            raise Exception((
                "Sample #{} has size (src={}, dst={}) but max size is {}."
                " Skip this example with --skip-invalid-size-inputs-valid-test"
            ).format(idx, src.sizes[idx], dst.sizes[idx], max_positions))

        sample_len = max(sample_len, src.sizes[idx], dst.sizes[idx])
        num_tokens = (len(batch) + 1) * sample_len
        if yield_batch(idx, num_tokens):
            yield batch
            batch = []
            sample_len = max(src.sizes[idx], dst.sizes[idx])

        batch.append(idx)

    if len(batch) > 0:
        yield batch

    # if len(ignored) > 0:
    #     print("Warning! {} samples are either too short or too long "
    #           "and will be ignored, first few sample ids={}".format(len(ignored), ignored[:10]))


def batches_by_size(src, dst, max_tokens=None, max_sentences=None,
                    max_positions=(1024, 1024), ignore_invalid_inputs=False,
                    descending=False):
    """Returns batches of indices sorted by size. Sequences with different
    source lengths are not allowed in the same batch."""
    assert isinstance(src, IndexedDataset) and isinstance(dst, IndexedDataset)
    if max_tokens is None:
        max_tokens = float('Inf')
    if max_sentences is None:
        max_sentences = float('Inf')
    indices = np.argsort(src.sizes, kind='mergesort')
    if descending:
        indices = np.flip(indices, 0)
    return list(_make_batches(
        src, dst, indices, max_tokens, max_sentences, max_positions,
        ignore_invalid_inputs, allow_different_src_lens=False))

def batches_by_size_test_classify(src, dst, max_tokens, max_sentences=None,
                    max_positions=(1024, 1024), ignore_invalid_inputs=False,
                    descending=False):
    """Returns batches of indices sorted by size. Sequences with different
    source lengths are not allowed in the same batch."""
    assert isinstance(src, IndexedDataset) and isinstance(dst, IndexedDataset)
    if max_tokens is None:
        max_tokens = float('Inf')
    if max_sentences is None:
        max_sentences = float('Inf')
    # max_positions = (2048,2048)
    print("src.sizes ", src.sizes)
    indices = np.argsort(src.sizes, kind='mergesort')
    if descending:
        indices = np.flip(indices, 0)
    print("type of indices:", type(indices))
    print(" shape of indices",indices.shape)
    print("indices:", indices)
    return list(_make_batches(
        src, dst, indices, max_tokens, max_sentences, max_positions,
        ignore_invalid_inputs, allow_different_src_lens=False))


def shuffled_batches_by_size(src, dst, max_tokens=None, max_sentences=None,
                             epoch=1, sample=0, max_positions=(1024, 1024),
                             sort_by_source_size=False):
    """Returns batches of indices, bucketed by size and then shuffled. Batches
    may contain sequences of different lengths."""
    assert isinstance(src, IndexedDataset) and isinstance(dst, IndexedDataset)
    if max_tokens is None:
        max_tokens = float('Inf')
    if max_sentences is None:
        max_sentences = float('Inf')

    if sample:
        indices = np.random.choice(len(src), sample, replace=False)
    else:
        indices = np.random.permutation(len(src))

    # sort by sizes
    indices = indices[np.argsort(dst.sizes[indices], kind='mergesort')]
    indices = indices[np.argsort(src.sizes[indices], kind='mergesort')]

    batches = list(_make_batches(
        src, dst, indices, max_tokens, max_sentences, max_positions,
        ignore_invalid_inputs=True, allow_different_src_lens=True))

    if not sort_by_source_size:
        np.random.shuffle(batches)

    # if sample:
    #     offset = (epoch - 1) * sample
    #     while offset > len(batches):
    #         np.random.shuffle(batches)
    #         offset -= len(batches)
    #
    #     result = batches[offset:(offset + sample)]
    #     while len(result) < sample:
    #         np.random.shuffle(batches)
    #         result += batches[:(sample - len(result))]
    #
    #     assert len(result) == sample, \
    #         "batch length is not correct {}".format(len(result))
    #
    #     batches = result

    return batches


# def mask_batches(batch_sampler, shard_id, num_shards):
#     if num_shards == 1:
#         return batch_sampler
#     res = [
#         batch
#         for i, batch in enumerate(batch_sampler)
#         if i % num_shards == shard_id
#     ]
#     expected_length = int(math.ceil(len(batch_sampler) / num_shards))
#     return res + [[]] * (expected_length - len(res))

# Custom mask_batches function to handle multi-gpu inconsistent batch sizes situation
import math

def mask_batches(batch_sampler, shard_id, num_shards, drop_last=True):
    print("batch_sampler: ", batch_sampler)
    print("type of batch_sampler: ", type(batch_sampler))
    print("drop_last: ", drop_last)
    if num_shards == 1:
        return batch_sampler
    res = [
        batch
        for i, batch in enumerate(batch_sampler)
        if i % num_shards == shard_id
    ]
    if drop_last and len(res) > 0 and len(res[-1]) < num_shards:
        res = res[:-1]  # Drop the last batch if it's incomplete
    return res


def mask_batches_test_classify(batch_sampler, shard_id, num_shards):
    if num_shards == 1:
        return batch_sampler
    res = [
        batch
        for i, batch in enumerate(batch_sampler)
        if i % num_shards == shard_id
    ]
    expected_length = int(math.ceil(len(batch_sampler) / num_shards))
    return res + [[]] * (expected_length - len(res))

@contextlib.contextmanager
def numpy_seed(seed):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)