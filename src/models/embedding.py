import sys, os, copy
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath("../"))

import numpy as np
import pandas as pd
import math


class SinCosPositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=300, device=torch.device('cpu')):
        super(SinCosPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = np.zeros((max_len, d_model))

        position = np.arange(0, max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        pe = torch.from_numpy(pe)
        pe = pe.unsqueeze(0).unsqueeze(1).float().to(device)
        self.register_buffer('pe', pe)

        self.d_model = d_model

    def forward(self, x):
        x = x + self.pe[:, :, :x.size(2), :] / np.sqrt(self.d_model)
        return self.dropout(x)


class RolePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=300, device=torch.device('cpu')):
        super(RolePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding(3, d_model)

        self.d_model = d_model

        self._reset_parameters()

    def forward(self, x, role_labels):
        x = x + self.embedding(role_labels) / np.sqrt(self.d_model)
        return self.dropout(x)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class DialogPositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=100, device=torch.device('cpu')):
        super(DialogPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding(max_len, d_model)

        self.d_model = d_model

        self._reset_parameters()

    def forward(self, x, position_labels):

        x = x + self.embedding(position_labels).unsqueeze(0) / np.sqrt(self.d_model)
        return self.dropout(x)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)