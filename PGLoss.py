import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math


class PGLoss(torch.nn.Module):
    
    def __init__(self, ignore_index=None, size_average=False, reduce=True):
        super(PGLoss, self).__init__()
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.reduce = reduce

    def forward(self, logprobs, label, reward, use_cuda):
        bsz, seqlen, _ = logprobs.size()
        loss = 0
        logprobs = logprobs.clone()
        print("logprobs shape:", logprobs.shape)
        for i in range(bsz):
            trg_label = label[i,:]
            print("trg_label shape:", trg_label.shape)
            print("PG forward i :", i)
            row_idx = torch.LongTensor(range(seqlen))
            if use_cuda:
                row_idx = row_idx.cuda()
            if self.ignore_index != None:
                logprobs[:, :, self.ignore_index] = 0
            
            print("row_idx shape:", row_idx.shape)
            # intermediate = logprobs[i, :, :][row_idx, trg_label]
            # print("Intermediate shape:", intermediate.shape)
            print("Reward[i]:", reward[i])

            print("NaN/Inf check:", torch.isnan(logprobs).any(), torch.isinf(logprobs).any(), torch.isnan(reward).any(), torch.isinf(reward).any())


            print("Shapes:", logprobs.shape, reward.shape)
            print("Indices:", i, row_idx, trg_label)
            print("Max/min indices values:", i.max().item(), i.min().item(), row_idx.max().item(), row_idx.min().item(), trg_label.max().item(), trg_label.min().item())

            # loss = loss + (-torch.sum(logprobs[i, :, :][row_idx, trg_label] * reward[i]))
            loss = loss + (-torch.sum(logprobs[i, row_idx, trg_label] * reward[i]))
        
        if self.size_average:
            loss = loss/bsz    

        
        return loss
