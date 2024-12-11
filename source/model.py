import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import math
import pickle



# custom learning rate
class LrStepTracker:
    def __init__(self, hidden_dim=512, warm_up_steps=4000, init_steps=0):
        self.hidden_dim = hidden_dim
        self.warm_up_steps = warm_up_steps
        self.init_steps = init_steps
        self.inverse_sqrt_dim = 1 / math.sqrt(hidden_dim)
        self.inverse_sqrt_warmup = 1 / (warm_up_steps * math.sqrt(warm_up_steps))
    
    def step(self, step):
        step += self.init_steps
        if step <= self.warm_up_steps:
            return self.inverse_sqrt_dim * self.inverse_sqrt_warmup * step
        else:
            inverse_sqrt_step = 1 / math.sqrt(step)
            return self.inverse_sqrt_dim * inverse_sqrt_step
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
        
# sinusoidal positional encoding like in attention is all you need, this will allow the transformer to see which token comes first

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, dropout=0.1, max_length=5000):
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) 
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# PADDING
    

# testing size and shape of data.
# with open("/home/bobert11/Desktop/474_final/MusicTransformer-Pytorch/dataset/e_piano/train/MIDI-UNPROCESSED_01-03_R1_2014_MID--AUDIO_03_R1_2014_wav--5.midi.pickle", "rb") as f:
#     data = pickle.load(f)
# print(type(data), len(data))


class TransformerModel(nn.Module):
    
    def __init__(self, vocab_size, embed_size=512, num_heads=8, num_layers=6, ff_hidden=1024, max_seq_len=2048, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size, dropout, max_seq_len)
        self.transformer = nn.Transformer(d_model=embed_size, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers, dim_feedforward=ff_hidden, dropout=0.1)
        self.fc_out = nn.Linear(embed_size, vocab_size)

        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        
        output = self.transformer(src.permute(1, 0, 2), tgt.permute(1, 0, 2), src_mask=src_mask, tgt_mask=tgt_mask)
        output = self.fc_out(output.permute(1, 0, 2))
        return output
