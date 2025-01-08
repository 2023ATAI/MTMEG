import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiheadSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.15):
        super(MultiheadSelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert self.head_dim * num_heads == hidden_dim, "Hidden dimension must be divisible by the number of heads"

        self.fc_query = nn.Linear(hidden_dim, hidden_dim)
        self.fc_key = nn.Linear(hidden_dim, hidden_dim)
        self.fc_value = nn.Linear(hidden_dim, hidden_dim)

        self.fc_out = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, src, mask=None):
        batch_size = src.shape[0]

        # Linear projections
        Q = self.fc_query(src)
        share = Q.data
        K = self.fc_key(src)
        V = self.fc_value(src)

        # Splitting heads
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1,
                                                                          3)  # (batch_size, num_heads, seq_len, head_dim)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention scores
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))

        # Attention weights
        attention = F.softmax(energy, dim=-1)

        # Attention dropout
        attention = self.dropout(attention)

        # Weighted sum
        x = torch.matmul(attention, V)

        # Concatenate heads
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.hidden_dim)

        # Final linear layer
        x = self.fc_out(x)

        return x,share


class PositionwiseFeedforward(nn.Module):
    def __init__(self, hidden_dim, pf_dim, dropout=0.1):
        super(PositionwiseFeedforward, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, pf_dim)
        self.fc2 = nn.Linear(pf_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class backbone(nn.Module):
    def __init__(self, inputsize, hinddensize):
        super(backbone, self).__init__()
        self.bone1 = nn.Conv1d(in_channels=inputsize,out_channels=hinddensize,kernel_size=3)
        self.bone2 = nn.Conv1d(in_channels=hinddensize,out_channels=hinddensize * 2,kernel_size=3)
        self.bone3 = nn.Conv1d(in_channels=hinddensize * 2,out_channels=hinddensize,kernel_size=3)
    def forward(self,x):
        src = x.permute(0,2,1)
        src = self.bone1(src)
        src = self.bone2(src)
        src = self.bone3(src)

        return src.permute(0,2,1)

class TransformerLayer(nn.Module):
    def __init__(self,inputsize, hidden_dim, num_heads, pf_dim, dropout=0.15):
        super(TransformerLayer, self).__init__()
        self.self_attn = MultiheadSelfAttention(hidden_dim, num_heads, dropout)
        self.ffn = PositionwiseFeedforward(hidden_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        # self.layer_norm1 = nn.Linear(14,hidden_dim)
        # self.layer_norm2 = nn.Linear(hidden_dim,hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.backbone =backbone(inputsize,hidden_dim)

    def positional_encoding(self,max_len, d_model):
        pos_enc = torch.zeros((1, max_len, d_model))
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc[0, :, 0::2] = torch.sin(pos * div_term)
        pos_enc[0, :, 1::2] = torch.cos(pos * div_term)
        return pos_enc

    def forward(self, cfg,src):
        device = torch.device(cfg['device']) if torch.cuda.is_available() else torch.device('cpu')
        # Self attention
        # _src = self.layer_norm1(src)
        # local_src = self.backbone(src)
        # src = self.layer_norm1(src)


        # 位置编码
        pos_encoding = self.positional_encoding(src.shape[1],src.shape[2]).to(src.device)
        src = src + pos_encoding

        # 将变量投影成三个空间
        _src = self.norm1(src)
        attn_output,share = self.self_attn(_src)
        # attn_output = self.norm(attn_output)

        src = src + self.dropout(attn_output)

        # Positionwise feedforward
        _src = self.norm2(src)
        ffn_output = self.ffn(_src)
        src = src + ffn_output

        return src,share

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, inputsize, hidden_dim, num_heads, pf_dim, dropout=0.15):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerLayer(inputsize, hidden_dim, num_heads, pf_dim, dropout) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, cfg, src):
        for i in range(self.num_layers):
            src,share = self.layers[i](cfg, src)
        return src,share