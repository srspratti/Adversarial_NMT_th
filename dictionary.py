# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch
import re


class Dictionary(object):
    """A mapping from symbols to consecutive integers"""
    def __init__(self, pad='<pad>', eos='</s>', unk='<unk>'):
        self.unk_word, self.pad_word, self.eos_word = unk, pad, eos
        self.symbols = []
        self.count = []
        self.indices = {}
        # dictionary indexing starts at 1 for consistency with Lua
        self.add_symbol('<Lua heritage>')
        self.pad_index = self.add_symbol(pad)
        self.eos_index = self.add_symbol(eos)
        self.unk_index = self.add_symbol(unk)
        self.nspecial = len(self.symbols)

    def __getitem__(self, idx):
        if idx < len(self.symbols):
            return self.symbols[idx]
        return self.unk_word

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return len(self.symbols)

    def index(self, sym):
        """Returns the index of the specified symbol"""
        # print("self.indices", self.indices)
        if sym in self.indices:
            return self.indices[sym]
        return self.unk_index

    def string(self, tensor, bpe_symbol=None, escape_unk=False):
        """Helper for converting a tensor of token indices to a string.

        Can optionally remove BPE symbols or escape <unk> words.
        """
        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return '\n'.join(self.string(t) for t in tensor)

        def token_string(i):
            if i == self.unk():
                return self.unk_string(escape_unk)
            elif i == self.pad():
                return ''
            else:
                return self[i]

        # find the (first) EOS token
        eos_idx = (tensor == self.eos()).nonzero()[0] if len((tensor == self.eos()).nonzero()) > 0 else -1
        # ignore the tokens after EOS
        tensor = tensor[:eos_idx]
        # obtain the raw words
        sent = ' '.join(token_string(i) for i in tensor if i != self.eos())

        if bpe_symbol is not None:
            sent = sent.replace(bpe_symbol, '')
        return sent

    def unk_string(self, escape=False):
        """Return unknown string, optionally escaped as: <<unk>>"""
        if escape:
            return '<{}>'.format(self.unk_word)
        else:
            return self.unk_word

    def add_symbol(self, word, n=1):
        """Adds a word to the dictionary"""
        if word in self.indices:
            idx = self.indices[word]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)
            self.count.append(n)
            return idx

    def finalize(self):
        """Sort symbols by frequency in descending order, ignoring special ones."""
        self.count, self.symbols = zip(
            *sorted(zip(self.count, self.symbols),
                    key=(lambda x: math.inf if self.indices[x[1]] < self.nspecial else x[0]),
                    reverse=True)
        )

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.pad_index

    def eos(self):
        """Helper to get index of end-of-sentence symbol"""
        return self.eos_index

    def unk(self):
        """Helper to get index of unk symbol"""
        return self.unk_index
    
    def unbpe(self, text):
        """
        Un-apply BPE encoding from a text.

        Args:
            text (str): BPE-encoded text.

        Returns:
            str: Text with BPE encoding removed.
        """
        # Using regex to replace instances of "@@ " with an empty string
        # print("before unbpe ", text)
        # text_unbpe = re.sub(r'@@ ?', '', text)
        return re.sub(r'@@ ?', '', text)
        # print("after unbpe ", text_unbpe)
        # return text_unbpe
    
    def ids_to_sentences(self, src_tokens):
        """
        Converts a 2D tensor of token IDs into a list of sentences.

        Args:
            src_tokens (torch.Tensor): A 2D tensor of token IDs.

        Returns:
            list: A list of sentences represented as strings.
        """
        # Ensure src_tokens is on the CPU and then convert to a list of lists
        src_tokens_list = src_tokens.cpu().numpy().tolist()

        sentences = []
        for ids in src_tokens_list:
            words = [self.__getitem__(idx) for idx in ids if idx not in [self.eos_index, self.pad_index]]
            sentence = ' '.join(words)
            # Un-apply BPE encoding from each sentence
            sentences.append(self.unbpe(sentence))
        return sentences 
    
    def sentences_to_ids(self, padded_bpe_translations, max_len=None):
        # Determine the maximum length if not provided
        if max_len is None:
            max_len = max(len(sentence.split()) for sentence in padded_bpe_translations)

        # Pre-allocate the tensor
        all_ids = torch.full((len(padded_bpe_translations), max_len), self.pad_index, dtype=torch.long)

        # Use the internal index mapping of the dictionary
        for idx, sentence in enumerate(padded_bpe_translations):
            words = sentence.split()
            ids = [self.indices.get(word, self.unk_index) for word in words]
            ids_tensor = torch.LongTensor(ids[:max_len])  # Truncate if needed
            all_ids[idx, :ids_tensor.size(0)] = ids_tensor

        return all_ids

    @staticmethod
    def load(f):
        """Loads the dictionary from a text file with the format:

        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```
        """

        if isinstance(f, str):
            try:
                with open(f, 'r', encoding='utf-8') as fd:
                    return Dictionary.load(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except Exception:
                raise Exception("Incorrect encoding detected in {}, please "
                                "rebuild the dataset".format(f))

        d = Dictionary()
        for line in f.readlines():
            idx = line.rfind(' ')
            word = line[:idx]
            count = int(line[idx+1:])
            d.indices[word] = len(d.symbols)
            d.symbols.append(word)
            d.count.append(count)
        return d

    def save(self, f, threshold=3, nwords=-1):
        """Stores dictionary into a text file"""
        if isinstance(f, str):
            with open(f, 'w', encoding='utf-8') as fd:
                return self.save(fd, threshold, nwords)
        cnt = 0
        for i, t in enumerate(zip(self.symbols, self.count)):
            if i >= self.nspecial and t[1] >= threshold \
                    and (nwords < 0 or cnt < nwords):
                print('{} {}'.format(t[0], t[1]), file=f)
                cnt += 1
