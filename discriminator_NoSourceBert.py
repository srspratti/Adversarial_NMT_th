import torch
import torch.nn as nn
# import torch.nn.functional as F
from transformers import BertModel, BertTokenizer


class Discriminator_NoSourceBert(nn.Module):
    def __init__(self, args, dst_dict, use_cuda = True):
        super(Discriminator_NoSourceBert, self).__init__()

        # below has been commented for now, but need to look into it
        # self.trg_dict_size = len(dst_dict)
        # self.pad_idx = dst_dict.pad()
  
        self.fixed_max_len = args.fixed_max_len
        self.use_cuda = use_cuda
        
        # Load pre-trained BERT model
        bert_model_name = "bert-base-uncased"
        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_embedding_dim = self.bert.config.hidden_size

        # self.embed_trg_tokens = Embedding(len(dst_dict), args.decoder_embed_dim, dst_dict.pad())


        self.conv1 = nn.Sequential(
            Conv2d(in_channels=bert_embedding_dim,
                   out_channels=512,
                   kernel_size=3,
                   stride=1,
                   padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            Conv2d(in_channels=bert_embedding_dim, # Not sure if we need to include 512 or bert_embedding_dim, change later to 512 if necessary
                   out_channels=256,
                   kernel_size=3,
                   stride=1,
                   padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            Linear(256 * 12 * 12, 20),
            nn.ReLU(),
            nn.Dropout(),
            Linear(20, 20),
            nn.ReLU(),
            Linear(20, 1),
        )

    def forward(self, trg_sentence):
        batch_size = trg_sentence.size(0)

        print("batch_size ", batch_size)
        # trg_out = self.embed_trg_tokens(trg_sentence)
        
        with torch.no_grad():
            trg_out = self.bert(trg_sentence)[0]
        
        print("trg_out type ", type(trg_out))
        print("trg_out size ", trg_out.size())
        
        out = trg_out.unsqueeze(2)
        out = out.permute(0,3,1,2)
        
        print("out type after out.permute(0,3,1,2) ", type(out))
        print("out size after out.permute(0,3,1,2) ", out.size())
        
        out = self.conv1(out)
        
        print("out type after self.conv1(out) ", type(out))
        print("out size after self.conv1(out) ", out.size())
        
        out = self.conv2(out)
        
        print("out type after self.conv2(out) ", type(out))
        print("out size after self.conv2(out) ", out.size())
        
        out = out.permute(0, 2, 3, 1)
        
        print("out type after out.permute(0, 2, 3, 1) ", type(out))
        print("out size after out.permute(0, 2, 3, 1) ", out.size())
        
        out = out.contiguous().view(batch_size, -1)
       
        print("out type after out.continguous", type(out))
        print("out size after out.continguous", out.size())
       
        out = torch.sigmoid(self.classifier(out))

        return out

def Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, **kwargs):
    m = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name:
            # param.data.uniform_(-0.1, 0.1)
            nn.init.kaiming_uniform_(param.data)
        elif 'bias' in name:
            nn.init.constant_(param.data, 0)
    return m

def Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, **kwargs):
    m = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name:
            # param.data.uniform_(-0.1, 0.1)
            nn.init.kaiming_uniform_(param.data)
        elif 'bias' in name:
            nn.init.constant_(param.data, 0)
    return m


def Linear(in_features, out_features, bias=True, dropout=0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    nn.init.kaiming_uniform_(m.weight.data)
    if bias:
        nn.init.constant_(m.bias.data, 0)
    return m


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.uniform_(-0.1, 0.1)
    return m