import torch
from torch import nn
from transformers import BertModel
import torch.nn.functional as F

"""
class Discriminator_cnn_bert_old(nn.Module):
    def __init__(self, args, use_cuda=True):
        super(Discriminator_cnn_bert, self).__init__()

        self.use_cuda = use_cuda
        self.args = args
        
        # bert_model_name_src='bert-base-uncased'
        bert_model_name_src='bert-base-multilingual-cased'
        bert_model_name_trg='bert-base-multilingual-cased'
        
        # Load pre-trained BERT models
        self.bert_src = BertModel.from_pretrained(bert_model_name_src)
        self.bert_trg = BertModel.from_pretrained(bert_model_name_trg)
        bert_embedding_dim = self.bert_src.config.hidden_size

        # Define the CNN layers
        self.conv1 = nn.Conv1d(in_channels=bert_embedding_dim, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)

        # Define the classifier
        self.classifier = nn.Linear(256, 1)
    
    # def forward(self, sentence):
    #     # Generate BERT embeddings
    #     with torch.no_grad():
    #         bert_out = self.bert(sentence)[0]

    #     # Pass embeddings through CNN layers
    #     conv1_out = F.relu(self.conv1(bert_out))
    #     conv2_out = F.relu(self.conv2(conv1_out))

    #     # Flatten the output and pass through the classifier
    #     flattened = conv2_out.view(conv2_out.size(0), -1)
    #     output = torch.sigmoid(self.classifier(flattened))

    #     return output
    
    def forward(self, src_sentence, trg_sentence):
        
        batch_size = src_sentence.size(0)
        print("batch_size ", batch_size)
        # Generate BERT embeddings for source and target sentences
        # Generate BERT embeddings without computing gradients
        with torch.no_grad():
            src_embed = self.bert_src(src_sentence)[0]  # Shape: [batch_size, seq_len, bert_embedding_dim]
            trg_embed = self.bert_trg(trg_sentence)[0]  # Shape: [batch_size, seq_len, bert_embedding_dim]

        print("src_embed shape ", src_embed.shape)
        print("trg_embed shape ", trg_embed.shape)
        
        # Combine embeddings (You might need to adjust this based on what you want to achieve)
        combined_embed = torch.cat([src_embed, trg_embed], dim=1)  # Shape: [batch_size, 2*seq_len, bert_embedding_dim]
        print("combined_embed shape ", combined_embed.shape)
        # Reshape for Conv2d (treat embedding_dim as channel dimension for Conv2d)
        # combined_embed = combined_embed.unsqueeze(1)  # Shape: [batch_size, 1, 2*seq_len, bert_embedding_dim]
        
        combined_embed = combined_embed.permute(0, 2, 1)  # Shape: [batch_size, bert_embedding_dim, 2*seq_len]
        print("combined_embed shape after permute ", combined_embed.shape)
        # Pass embeddings through CNN layers
        conv1_out = F.relu(self.conv1(combined_embed))  # Apply Conv2d
        print("conv1_out shape ", conv1_out.shape)
        conv2_out = F.relu(self.conv2(conv1_out))
        print("conv2_out shape ", conv2_out.shape)
        
        # # Flatten and pass through classifier
        # output = self.classifier(conv2_out)
        # print("shape of output ", output.shape)
        
        # output = output.contiguous().view(batch_size, -1)
        # print("shape of output after contiguous ", output.shape)
        # output = torch.sigmoid(self.classifier(output))
        # print("shape of output after sigmoid ", output.shape)
        
        # Assuming conv2_out is [batch_size, channels, length], flatten for classifier
        flattened = conv2_out.view(conv2_out.size(0), -1)
        print("shape of flattened ", flattened.shape)
        
        # Apply classifier once correctly
        output = self.classifier(flattened)
        print("shape of output ", output.shape)
        output = torch.sigmoid(output)  # Ensure output is between [0, 1]
        print("shape of output after sigmoid: ", output.shape)
        
        # Ensure output is correctly sized [batch_size, 1]
        # print("shape of output ", output.shape)
        
        return output
        
        """

class Discriminator_cnn_bert(nn.Module):
    def __init__(self, args, use_cuda=True):
        super(Discriminator_cnn_bert, self).__init__()

        self.use_cuda = use_cuda
        self.args = args
        
        # bert_model_name_src='bert-base-uncased'
        bert_model_name_src='bert-base-multilingual-cased'
        bert_model_name_trg='bert-base-multilingual-cased'
        
        
        # Load pre-trained BERT models for source and target
        self.bert_src = BertModel.from_pretrained(bert_model_name_src)
        self.bert_trg = BertModel.from_pretrained(bert_model_name_trg)
        
        bert_embedding_dim = self.bert_src.config.hidden_size
        
        self.conv1 = nn.Sequential(
            Conv2d(in_channels=bert_embedding_dim * 2,  # Because we're stacking the embeddings,
                   out_channels=512,
                   kernel_size=3,
                   stride=1,
                   padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            Conv2d(in_channels=512,
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
            Linear(256 * 32 * 32, 20),
            nn.ReLU(),
            nn.Dropout(),
            Linear(20, 20),
            nn.ReLU(),
            Linear(20, 1),
        )


        """
        # Adjust the in_channels for Conv2d based on BERT output and concatenation
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=bert_embedding_dim * 2,  # Because we're stacking the embeddings
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=512,
                      out_channels=256,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        # The classifier part might need adjustment based on the output size after conv layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(256 *12*12, 20),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(20, 1),
            nn.Sigmoid()
        )
        """
        
        
        
    # def forward_old(self, src_sentence, trg_sentence):
    #     # Generate BERT embeddings for source and target sentences without computing gradients
    #     with torch.no_grad():
    #         src_embed = self.bert_src(src_sentence)[0]  # Shape: [batch_size, seq_len, bert_embedding_dim]
    #         trg_embed = self.bert_trg(trg_sentence)[0]  # Shape: [batch_size, seq_len, bert_embedding_dim]
    #     print("src_embed shape ", src_embed.shape)
    #     print("trg_embed shape ", trg_embed.shape)
    #     # Stack embeddings along the embedding dimension to match the in_channels for Conv2d
    #     combined_embed = torch.cat((src_embed, trg_embed), dim=-1).unsqueeze(1)  # Adding a channel dim
    #     print("combined_embed shape ", combined_embed.shape)

    #     out = self.conv1(combined_embed)
    #     print("conv1 out shape ", out.shape)
    #     out = self.conv2(out)
    #     print("conv2 out shape ", out.shape)

    #     # Pass through the classifier
    #     out = self.classifier(out)
    #     print("classifier out shape ", out.shape)

    #     return out
    
    def forward(self, src_sentence, trg_sentence):
            batch_size = src_sentence.size(0)

            print("batch_size ", batch_size)
            src_out = self.bert_src(src_sentence)[0]  # Shape: [batch_size, seq_len, bert_embedding_dim]
            trg_out = self.bert_trg(trg_sentence)[0]  # Shape: [batch_size, seq_len, bert_embedding_dim]

            print("src_out type ", type(src_out))
            print("src_out size ", src_out.size())
            
            print("trg_out type ", type(trg_out))
            print("trg_out size ", trg_out.size())
            
            src_out = torch.stack([src_out] * trg_out.size(1), dim=2)
            trg_out = torch.stack([trg_out] * src_out.size(1), dim=1)
            
            print("after torch.stack....")
            print("src_out type ", type(src_out))
            print("src_out size ", src_out.size())
            
            print("trg_out type ", type(trg_out))
            print("trg_out size ", trg_out.size())
            
            out = torch.cat([src_out, trg_out], dim=3)
            
            print("out type after torch.cat ", type(out))
            print("out size after torch.cat ", out.size())
            
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
            
            print("out shape returning from the discriminator:" , out.shape)
            print("out type returning from the discriminator:" , type(out))
            print("out returning from the discriminator:" , out)
            
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