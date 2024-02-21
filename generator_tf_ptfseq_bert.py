import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import utils


# new S15 

# import torch
# from torch import nn
# import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer
from transformers import BertModel

class TransformerModel_custom_bert(nn.Module):
    def __init__(self, args, src_dict, dst_dict, use_cuda=True):
        super(TransformerModel_custom_bert, self).__init__()
        self.args = args
        self.use_cuda = use_cuda
        self.src_dict = src_dict
        self.dst_dict = dst_dict
        
        # SEP changes
        print("length of source dictionary: ", len(src_dict))
        print("length of dst dictionary: ", len(dst_dict))
        print("args.encoder_embed_dim) ", args.encoder_embed_dim)
        
        self.src_bert_embed = BertModel.from_pretrained('bert-base-uncased')
        self.tgt_bert_embed = BertModel.from_pretrained('bert-base-uncased')
    
        self.encoder = TransformerEncoder(TransformerEncoderLayer(d_model=args.encoder_embed_dim, nhead=args.encoder_heads, dim_feedforward=args.encoder_ffn_embed_dim, dropout=args.dropout), args.encoder_layers)
        self.decoder = TransformerDecoder(TransformerDecoderLayer(d_model=args.decoder_embed_dim, nhead=args.decoder_heads, dim_feedforward=args.decoder_ffn_embed_dim, dropout=args.dropout), args.decoder_layers)

        self.out = nn.Linear(args.decoder_embed_dim, len(dst_dict))
        

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, sample, args):
        
        # print(" sample shape: ", sample)
        # print(" sample keys: ", sample.keys())
        
        src = sample['net_input']['src_tokens']
        
        # print(" src shape: ", src.shape)
        
        # Should it be removed or kept ? For now, we are removing it
        # src = src.permute(1, 0)  # The transformer expects seq_len, batch as input dimensions
        
        # Create an attention mask for BERT: '1' for real tokens, '0' for padding
        src_attention_mask = (src != 0)

        # Use BERT for source embedding
        encoded_src = self.src_bert_embed(src, attention_mask=src_attention_mask)
        encoder_out = self.encoder(encoded_src.last_hidden_state)
    
        tgt = sample['net_input']['prev_output_tokens']
        # print("tgt before permute shape, ", tgt.shape)
        
        # Should it be removed or kept ? For now, we are removing it
        # tgt = tgt.permute(1, 0)
        
        tgt_attention_mask = (tgt != 0)
        
        # tgt_mask = self.generate_square_subsequent_mask(len(tgt)).to(tgt.device)
        
        # print("tgt_mask shape, ", tgt_mask.shape)

        # embedded_tgt = self.tgt_embed(tgt)
        # Use BERT for target embedding
        encoded_tgt = self.tgt_bert_embed(tgt, attention_mask=tgt_attention_mask)
        
        tgt_mask = self.generate_square_subsequent_mask(len(tgt)).to(tgt.device)
        decoder_out = self.decoder(
        encoded_tgt.last_hidden_state.permute(1,0,2), 
        encoder_out.permute(1,0,2)
        )
        
        print("decoder_out shape, ", decoder_out.shape)
        
        output = self.out(decoder_out.permute(1, 0, 2))
        
        print("output shape, ", output.shape)
        
        print("F.log_softmax(output, dim=2) ", F.log_softmax(output, dim=2).shape)
        
        return F.log_softmax(output, dim=2)

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""
        vocab = net_output.size(-1)
        net_output1 = net_output.view(-1, vocab)
        if log_probs:
            return F.log_softmax(net_output1, dim=1).view_as(net_output)
        else:
            return F.softmax(net_output1, dim=1).view_as(net_output)

# To-Do : decide to keep the below methods or not ? 

def Embedding(num_embeddings, embedding_dim, padding_idx):
    print("num_embeddings in def Embedding: ", num_embeddings)
    print("embedding_dim in def Embedding: ", num_embeddings)
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.uniform_(-0.1, 0.1)
    return m


def LSTM(input_size, hidden_size, **kwargs):
    m = nn.LSTM(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def LSTMCell(input_size, hidden_size, **kwargs):
    m = nn.LSTMCell(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def Linear(in_features, out_features, bias=True, dropout=0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m

