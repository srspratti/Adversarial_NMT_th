import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import utils

import torch
import torch.nn as nn
import torch.nn.functional as F

# class Transformer(nn.Module):
#     def __init__(self, args, src_dict, tgt_dict, embed_dim, num_heads, hidden_dim, num_layers, dropout, use_cuda=True):
#         super().__init__()
#         self.embed_dim = args.encoder_embed_dim
#         self.num_heads = args.num_heads
#         self.use_cuda = use_cuda
#         self.embed_src = nn.Embedding(len(src_dict), embed_dim, padding_idx=src_dict['<pad>'])
#         self.embed_tgt = nn.Embedding(len(tgt_dict), embed_dim, padding_idx=tgt_dict['<pad>'])
#         self.pos_enc = PositionalEncoding(embed_dim, dropout=dropout)
#         self.encoder_layers = nn.ModuleList([EncoderLayer(embed_dim, num_heads, hidden_dim, dropout) for _ in range(num_layers)])
#         self.decoder_layers = nn.ModuleList([DecoderLayer(embed_dim, num_heads, hidden_dim, dropout) for _ in range(num_layers)])
#         self.generator = nn.Linear(embed_dim, len(tgt_dict))
        
#     def forward(self, src, tgt):
#         # Encode the source sentence
#         src_emb = self.embed_src(src) * math.sqrt(self.embed_dim)
#         src_emb = self.pos_enc(src_emb)
#         memory = self.encode(src_emb)
        
#         # Decode the target sentence
#         tgt_emb = self.embed_tgt(tgt) * math.sqrt(self.embed_dim)
#         tgt_emb = self.pos_enc(tgt_emb)
#         output = self.decode(memory, tgt_emb)
#         return self.generator(output)
    
#     def encode(self, src_emb):
#         mask = get_pad_mask(src_emb, src_emb)
#         output = src_emb
#         for layer in self.encoder_layers:
#             output = layer(output, mask)
#         return output
    
#     def decode(self, memory, tgt_emb):
#         tgt_mask = get_subsequent_mask(tgt_emb)
#         memory_mask = get_pad_mask(tgt_emb, memory)
#         output = tgt_emb
#         for layer in self.decoder_layers:
#             output = layer(output, memory, tgt_mask, memory_mask)
#         return output
    
# class PositionalEncoding(nn.Module):
#     def __init__(self, embed_dim, dropout, max_len=5000):
#         super().__init__()
#         self.dropout = nn.Dropout(dropout)
#         pe = torch.zeros(max_len, embed_dim)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)
        
#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)
    
# class EncoderLayer(nn.Module):
#     def __init__(self, embed_dim, num_heads, hidden_dim, dropout):
#         super().__init__()
#         self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
#         self.feedforward = nn.Sequential(
#             nn.Linear(embed_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, embed_dim),
#             nn.Dropout(dropout),
#         )
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.norm2 = nn.LayerNorm(embed_dim)
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, x, mask):
#         x = self.norm1(x)
#         attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
#         x = x + self.dropout(attn_output)
#         x = self.norm2(x)
#         ff_output = self.feedforward(x)
#         x = x + self.dropout(ff_output)
#         return x

# class DecoderLayer(nn.Module):
#     def __init__(self, embed_dim, num_heads, hidden_dim, dropout):
#         super().__init__()
#         self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
#         self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
#         self.feedforward = nn.Sequential(
#             nn.Linear(embed_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, embed_dim),
#             nn.Dropout(dropout),
#         )
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.norm2 = nn.LayerNorm(embed_dim)
#         self.norm3 = nn.LayerNorm(embed_dim)
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, x, memory, tgt_mask, memory_mask):
#         x = self.norm1(x)
#         self_attn_output, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
#         x = x + self.dropout(self_attn_output)
#         x = self.norm2(x)
#         cross_attn_output, _ = self.cross_attn(x, memory, memory, attn_mask=memory_mask)
#         x = x + self.dropout(cross_attn_output)
#         x = self.norm3(x)
#         ff_output = self.feedforward(x)
#         x = x + self.dropout(ff_output)
#         return x

# def get_pad_mask(seq, pad_idx):
#     return (seq != pad_idx).unsqueeze(-2)

# def get_subsequent_mask(seq):
#     seq_len = seq.size(1)
#     subsequent_mask = torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1)
#     return subsequent_mask == 0

# new S15 

# import torch
# from torch import nn
# import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer

class TransformerModel_check(nn.Module):
    def __init__(self, args, src_dict, dst_dict, use_cuda=True):
        super(TransformerModel, self).__init__()
        self.args = args
        self.use_cuda = use_cuda
        self.src_dict = src_dict
        self.dst_dict = dst_dict

        self.encoder_embedding = nn.Embedding(len(src_dict), args.encoder_embed_dim)
        self.decoder_embedding = nn.Embedding(len(dst_dict), args.decoder_embed_dim)

        self.transformer_encoder = TransformerEncoder(TransformerEncoderLayer(d_model=args.encoder_embed_dim, nhead=args.encoder_heads, dim_feedforward=args.encoder_ffn_embed_dim, dropout=args.dropout), args.encoder_layers)
        self.transformer_decoder = TransformerDecoder(TransformerDecoderLayer(d_model=args.decoder_embed_dim, nhead=args.decoder_heads, dim_feedforward=args.decoder_ffn_embed_dim, dropout=args.dropout), args.decoder_layers)

        self.out = nn.Linear(args.decoder_embed_dim, len(dst_dict))

    def forward(self, src, tgt):
        src_emb = self.encoder_embedding(src)
        tgt_emb = self.decoder_embedding(tgt)
        memory = self.transformer_encoder(src_emb.permute(1, 0))
        output = self.transformer_decoder(tgt_emb.permute(1, 0), memory)
        output = self.out(output.permute(1, 0))
        return F.log_softmax(output, dim=2)

class TransformerModel_custom(nn.Module):
    def __init__(self, args, src_dict, dst_dict, use_cuda=True):
        super(TransformerModel_custom, self).__init__()
        self.args = args
        self.use_cuda = use_cuda
        self.src_dict = src_dict
        self.dst_dict = dst_dict
        
        # SEP changes
        print("length of source dictionary: ", len(src_dict))
        print("length of dst dictionary: ", len(dst_dict))
        print("args.encoder_embed_dim) ", args.encoder_embed_dim)
        self.src_embed = nn.Embedding(len(src_dict), args.encoder_embed_dim)
        self.tgt_embed = nn.Embedding(len(dst_dict), args.decoder_embed_dim)
        

        print("src_embed ", self.src_embed)
        print("src_embed ", self.src_embed)
        print("src_embed type: ", type(self.src_embed))
        print("tgt_embed type: ", type(self.tgt_embed))
        
        # Initialize encoder and decoder
        # self.encoder = nn.TransformerEncoder(
        #     nn.Embedding(len(src_dict), args.encoder_embed_dim), 
        #     num_layers=args.encoder_layers
        #     # dropout=args.encoder_dropout_in, # To-Do : currently added encoder_dropout_in but need to check should it be in or out? 
        # )

        # self.decoder = nn.TransformerDecoder(
        #     nn.Embedding(len(dst_dict), args.decoder_embed_dim), 
        #     num_layers=args.decoder_layers
        #     # dropout=args.decoder_dropout_in,
        # )
        
        # self.encoder = nn.TransformerEncoder(
        #     self.src_embed, 
        #     num_layers=args.encoder_layers
        #     # dropout=args.encoder_dropout_in, # To-Do : currently added encoder_dropout_in but need to check should it be in or out? 
        # )
        
        self.encoder = TransformerEncoder(TransformerEncoderLayer(d_model=args.encoder_embed_dim, nhead=args.encoder_heads, dim_feedforward=args.encoder_ffn_embed_dim, dropout=args.dropout), args.encoder_layers)
        self.decoder = TransformerDecoder(TransformerDecoderLayer(d_model=args.decoder_embed_dim, nhead=args.decoder_heads, dim_feedforward=args.decoder_ffn_embed_dim, dropout=args.dropout), args.decoder_layers)

        self.out = nn.Linear(args.decoder_embed_dim, len(dst_dict))
        
        # self.decoder = nn.TransformerDecoder(
        #     self.tgt_embed, 
        #     num_layers=args.decoder_layers
        #     # dropout=args.decoder_dropout_in,
        # )

    # def forward(self, sample):
    #     src = sample['net_input']['src_tokens']
    #     print("src type :", type(src))
    #     print("src shape : ", src.shape)
    #     src = src.permute(1, 0)  # The transformer model expects seq_len, batch as input dimensions
    #     encoder_out = self.encoder(src)

    #     tgt = sample['net_input']['prev_output_tokens']
    #     tgt = tgt.permute(1, 0)  # The transformer model expects seq_len, batch as input dimensions
    #     decoder_out = self.decoder(tgt, encoder_out)

    #     return decoder_out.permute(1, 0, 2)  # Permute back to batch, seq_len, dim for compatibility with the rest of the code

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    # def forward(self, sample):
    #     src = sample['net_input']['src_tokens']
    #     src = src.permute(1, 0)  # The transformer model expects seq_len, batch as input dimensions
    #     src_mask = self.generate_square_subsequent_mask(len(src)).to(src.device)
    #     encoder_out = self.encoder(self.src_embed(src), src_key_padding_mask=src_mask)

    #     tgt = sample['net_input']['prev_output_tokens']
    #     tgt = tgt.permute(1, 0)  # The transformer model expects seq_len, batch as input dimensions
    #     tgt_mask = self.generate_square_subsequent_mask(len(tgt)).to(tgt.device)
    #     decoder_out = self.decoder(self.tgt_embed(tgt), encoder_out, tgt_mask=tgt_mask)

    #     return decoder_out.permute(1, 0, 2)  # Permute back to batch, seq_len, dim for compatibility with the rest of the code
    
    def forward(self, sample, args):
        print(" sample shape: ", sample)
        print(" sample keys: ", sample.keys())
        src = sample['net_input']['src_tokens']
        print(" src shape: ", src.shape)
        src = src.permute(1, 0)  # The transformer expects seq_len, batch as input dimensions
        print(" src after permute shape: ", src.shape)
        
        # For simplicity, let's assume padding tokens are represented by the number 0
        src_padding_mask = (src == 0)
        print(" src_padding_mask shape: ", src_padding_mask.shape)
        
        # self.src_embed = nn.Embedding(len(src_dict), args.encoder_embed_dim)


        # embedded_src = nn.Embedding(src, args.encoder_embed_dim)
        embedded_src = self.src_embed(src)
        print(" embedded_src shape: ", embedded_src.shape)
        print("\n----------------------------------------------------------------")
        print(" embedded_src : ", embedded_src)
        print("\n----------------------------------------------------------------")
        #encoder_out = self.encoder(src=embedded_src.permute(1,0,2), src_key_padding_mask=src_padding_mask.permute(1,0))
        encoder_out = self.encoder(src=embedded_src)
    
        tgt = sample['net_input']['prev_output_tokens']
        print("tgt before permute shape, ", tgt.shape)
        tgt = tgt.permute(1, 0)
        print("tgt after permute shape, ", tgt.shape)

        tgt_padding_mask = (tgt == 0)
        print("tgt_padding_mask shape, ", tgt_padding_mask.shape)
        tgt_mask = self.generate_square_subsequent_mask(len(tgt)).to(tgt.device)
        print("tgt_mask shape, ", tgt_mask.shape)

        embedded_tgt = self.tgt_embed(tgt)
        print("embedded_tgt shape, ", embedded_tgt.shape)
        print("encoder_out shape, ", encoder_out.shape)
        print("src_padding_mask shape, ", src_padding_mask.shape)
        print("tgt_padding_mask shape, ", tgt_padding_mask.shape)
        # decoder_out = self.decoder(
        #     embedded_tgt, 
        #     encoder_out, 
        #     tgt_mask=tgt_mask, 
        #     memory_key_padding_mask=src_padding_mask, 
        #     tgt_key_padding_mask=tgt_padding_mask
        # )
        
        decoder_out = self.decoder(
            embedded_tgt.permute(1,0,2), 
            encoder_out.permute(1,0,2)
        )
        print("decoder_out shape, ", decoder_out.shape)
        
        output = self.out(decoder_out.permute(1, 0, 2))
        
        print("output shape, ", output.shape)
        
        print("F.log_softmax(output, dim=2) ", F.log_softmax(output, dim=2).shape)
        
        return F.log_softmax(output, dim=2)
    
        # return decoder_out.permute(1, 0, 2)  # Permute back to batch, seq_len, dim for compatibility
        # def get_normalized_probs(self, net_output, log_probs):
    #     """Get normalized probabilities (or log probs) from a net's output."""
    #     vocab = net_output.size(-1)
    #     net_output1 = net_output.view(-1, vocab)
    #     if log_probs:
    #         return F.log_softmax(net_output1, dim=1).view_as(net_output)
    #     else:
    #         return F.softmax(net_output1, dim=1).view_as(net_output)

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

