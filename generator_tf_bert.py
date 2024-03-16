import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizerFast

class TransformerModel_bert(nn.Module):
    def __init__(self, args, use_cuda=True):
        super(TransformerModel_bert, self).__init__()
        self.args = args
        self.use_cuda = use_cuda
        # self.bert_model = bert_model
        
        # Load BERT model for source and target embeddings
        # self.src_bert_embed = BertModel.from_pretrained('bert-base-uncased')
        # self.tgt_bert_embed = BertModel.from_pretrained('bert-base-uncased')
        # self.src_bert_embed = BertModel.from_pretrained(bert_model)
        # self.tgt_bert_embed = BertModel.from_pretrained(bert_model)
         # Load BERT model for source (English) and target (French) embeddings
        # self.src_bert_embed = BertModel.from_pretrained('bert-base-uncased')
        self.src_bert_embed = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.tgt_bert_embed = BertModel.from_pretrained('bert-base-multilingual-cased')
        
        # Define Transformer encoder and decoder
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=args.encoder_embed_dim, nhead=args.encoder_heads, dim_feedforward=args.encoder_ffn_embed_dim, dropout=args.dropout), args.encoder_layers)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=args.decoder_embed_dim, nhead=args.decoder_heads, dim_feedforward=args.decoder_ffn_embed_dim, dropout=args.dropout), args.decoder_layers)

        # Define output layer
        self.out = nn.Linear(args.decoder_embed_dim, self.tgt_bert_embed.config.vocab_size)

    def forward(self, src, tgt):
        # Create attention masks for BERT
        src_attention_mask = (src != 0)
        tgt_attention_mask = (tgt != 0)

        # Use BERT for source and target embeddings
        encoded_src = self.src_bert_embed(src, attention_mask=src_attention_mask).last_hidden_state
        encoded_tgt = self.tgt_bert_embed(tgt, attention_mask=tgt_attention_mask).last_hidden_state

        # Pass embeddings through Transformer encoder and decoder
        encoder_out = self.encoder(encoded_src)
        print("encoder_out shape: ", encoder_out.shape)
        decoder_out = self.decoder(encoded_tgt, encoder_out)
        print("decoder_out shape: ", decoder_out.shape)

        # Pass decoder output through final layer
        output = self.out(decoder_out)
        print("output shape: ", output.shape)
        log_probs = F.log_softmax(output, dim=-1)
        print("log_probs shape: ", log_probs.shape)

        return log_probs, decoder_out