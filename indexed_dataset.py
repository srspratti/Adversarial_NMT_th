# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import os
import struct
import torch

from tokenizer import Tokenizer


def read_longs(f, n):
    a = np.empty(n, dtype=np.int64)
    f.readinto(a)
    return a


def write_longs(f, a):
    f.write(np.array(a, dtype=np.int64))


dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float,
    7: np.double,
}


def code(dtype):
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k


class IndexedDataset(object):
    """Loader for TorchNet IndexedDataset"""

    def __init__(self, path):
        with open(path + '.idx', 'rb') as f:
            magic = f.read(8)
            print("magic: ",magic)
            print("file f: ", f)
            # assert magic == b'TNTIDX\x00\x00'
            # assert magic == b'MMIDIDX\x00'
            version = f.read(8)
            print("struct.unpack('<Q', version) :", struct.unpack('<Q', version))
            # assert struct.unpack('<Q', version) == (1,)
            assert struct.unpack('<Q', version) in [(1,), (256,)]
            code, self.element_size = struct.unpack('<QQ', f.read(16))
            # code, self.element_size = struct.unpack('>QQ', f.read(16))
            print("self.element_size ", self.element_size)
            print('Available dtypes:', dtypes.keys())
            print('Trying to access key:', code)
            self.dtype = dtypes[code]
            # if code in dtypes:
            #     self.dtype = dtypes[code]
            # else:
            #     print(f"Unexpected code: {code}. Expected one of {list(dtypes.keys())}.")
            self.size, self.s = struct.unpack('<QQ', f.read(16))
            self.dim_offsets = read_longs(f, self.size + 1)
            self.data_offsets = read_longs(f, self.size + 1)
            self.sizes = read_longs(f, self.s)
        self.read_data(path)

    def read_data(self, path):
        self.data_file = open(path + '.bin', 'rb', buffering=0)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError('index out of range')

    def __del__(self):
        self.data_file.close()

    def __getitem__(self, i):
        self.check_index(i)
        tensor_size = self.sizes[self.dim_offsets[i]:self.dim_offsets[i + 1]]
        a = np.empty(tensor_size, dtype=self.dtype)
        self.data_file.seek(self.data_offsets[i] * self.element_size)
        self.data_file.readinto(a)
        return torch.from_numpy(a)

    def __len__(self):
        return self.size

    @staticmethod
    def exists(path):
        return os.path.exists(path + '.idx')


class IndexedInMemoryDataset(IndexedDataset):
    """Loader for TorchNet IndexedDataset, keeps all the data in memory"""

    def read_data(self, path):
        self.data_file = open(path + '.bin', 'rb')
        self.buffer = np.empty(self.data_offsets[-1], dtype=self.dtype)
        self.data_file.readinto(self.buffer)
        self.data_file.close()

    def __del__(self):
        pass

    def __getitem__(self, i):
        self.check_index(i)
        tensor_size = self.sizes[self.dim_offsets[i]:self.dim_offsets[i + 1]]
        a = np.empty(tensor_size, dtype=self.dtype)
        np.copyto(a, self.buffer[self.data_offsets[i]:self.data_offsets[i + 1]])
        return torch.from_numpy(a)


class IndexedRawTextDataset(IndexedDataset):
    """Takes a text file as input and binarizes it in memory at instantiation.
    Original lines are also kept in memory"""

    def __init__(self, path, dictionary):
        self.tokens_list = []
        self.lines = []
        self.sizes = []
        self.read_data(path, dictionary)
        self.size = len(self.tokens_list)

    def read_data(self, path, dictionary):
        print("path ", path)
        with open(path, 'r') as f:
            for line in f:
                print("line : ", line)
                self.lines.append(line.strip('\n'))
                # +1 for Lua compatibility
                tokens = Tokenizer.tokenize(line, dictionary, add_if_not_exist=False) + 1
                print("tokens ", tokens)
                self.tokens_list.append(tokens)
                self.sizes.append(len(tokens))
        self.sizes = np.array(self.sizes)

    def __getitem__(self, i):
        self.check_index(i)
        return self.tokens_list[i]

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

class IndexedRawTextDataset_label(IndexedDataset):
    """Takes a text file containing labels as input and binarizes it in memory at instantiation.
    Original lines are also kept in memory"""

    def __init__(self, path):
        # self.tokens_list = []
        self.lines = []
        self.sizes = []
        self.read_data(path)
        self.size = len(self.sizes)

    def read_data(self, path):
        print("path ", path)
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                print("line : ", line)
                self.lines.append(int(line.strip('\n')))
                # # +1 for Lua compatibility
                # tokens = Tokenizer.tokenize(line, dictionary, add_if_not_exist=False) + 1
                # self.tokens_list.append(tokens)
                self.sizes.append(i)
        self.sizes = np.array(self.sizes)

    def __getitem__(self, i):
        self.check_index(i)
        return self.lines[i]

    def get_original_label(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size


class IndexedDatasetBuilder(object):

    element_sizes = {
        np.uint8: 1,
        np.int8:  1,
        np.int16: 2,
        np.int32: 4,
        np.int64: 8,
        np.float: 4,
        np.double: 8
    }

    def __init__(self, out_file, dtype=np.int32):
        self.out_file = open(out_file, 'wb')
        self.dtype = dtype
        self.data_offsets = [0]
        self.dim_offsets = [0]
        self.sizes = []
        self.element_size = self.element_sizes[self.dtype]

    def add_item(self, tensor):
        # +1 for Lua compatibility
        bytes = self.out_file.write(np.array(tensor.numpy() + 1, dtype=self.dtype))
        self.data_offsets.append(self.data_offsets[-1] + bytes / self.element_size)
        for s in tensor.size():
            self.sizes.append(s)
        self.dim_offsets.append(self.dim_offsets[-1] + len(tensor.size()))

    def finalize(self, index_file):
        self.out_file.close()
        index = open(index_file, 'wb')
        index.write(b'TNTIDX\x00\x00')
        index.write(struct.pack('<Q', 1))
        index.write(struct.pack('<QQ', code(self.dtype),
                                self.element_size))
        index.write(struct.pack('<QQ', len(self.data_offsets) - 1,
                                len(self.sizes)))
        write_longs(index, self.dim_offsets)
        write_longs(index, self.data_offsets)
        write_longs(index, self.sizes)
        index.close()
