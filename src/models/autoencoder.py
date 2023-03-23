import os, sys, copy

sys.path.append(os.path.abspath("../"))
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import logging, math

from models.graph import GAT
from models.decoder import GateTransformerDecoderLayer
from models.embedding import SinCosPositionalEncoding, RolePositionalEncoding
import utils, config

DEBUG = False


class AutoEncoderWithTransformer(nn.Module):

    def __init__(self, voc, params={}, device=torch.device('cpu')):
        super(AutoEncoderWithTransformer, self).__init__()

        self.device = device
        self.voc = voc

        self.d_model = params['d_model']
        self.nhead = params['nhead']
        self.window_size = params['window_size']
        self.dim_feedforward = params['dim_feedforward']
        self.activation = params['activation']
        self.dropout = params['dropout']
        self.utterance_encoder_layer_number = params['utterance_encoder_layer_number']
        self.utterance_decoder_layer_number = params['utterance_decoder_layer_number']
        self.max_dialogue_length = params['max_dialogue_length']
        self.max_utterance_length = params['max_utterance_length']

        # utterance encoder
        utterance_encoder_layer = nn.TransformerEncoderLayer(self.d_model, self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, activation=self.activation)
        utterance_encoder_norm = nn.LayerNorm(self.d_model)
        self.utterance_encoder = nn.TransformerEncoder(utterance_encoder_layer, self.utterance_encoder_layer_number, norm=utterance_encoder_norm)

        # utterance decoder
        utterance_decoder_layer = GateTransformerDecoderLayer(self.d_model, self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, activation=self.activation)
        utterance_decoder_norm = nn.LayerNorm(self.d_model)
        self.utterance_decoder = nn.TransformerDecoder(utterance_decoder_layer, self.utterance_decoder_layer_number, utterance_decoder_norm)

        # graph attention network GAT layer for dialog encoder
        self.gat_encoder = GAT(n_in_feat=self.d_model, n_hid=self.d_model, n_out_feat=self.d_model, dropout=self.dropout, alpha=0.2, nheads=self.nhead)

        self.embedding = nn.Embedding(len(self.voc), self.d_model)
        self.fc = nn.Linear(self.d_model, len(self.voc))

        self.position_embedding = SinCosPositionalEncoding(self.d_model, dropout=self.dropout, max_len=self.max_utterance_length)
        self.role_embedding = RolePositionalEncoding(self.d_model, dropout=self.dropout, max_len=self.max_dialogue_length)

    def encode(self, inputs={}):
        # dias: [batch, dia_max_len, utt_max_len]
        # dia_lens: [batch]
        # utt_lens: [batch, dia_max_len]

        dialogue = inputs['dialogue']
        role = inputs['role']
        utterance_padding_mask = inputs['utterance_padding_mask']

        dia_batch_size, dia_max_len, utt_max_len = dialogue.size()

        # token embedding
        dialogue = self.embedding(dialogue) * math.sqrt(self.d_model)
        dialogue = self.position_embedding(dialogue)

        ######################### Utterance Encoder #################################
        utterance_padding_mask = utterance_padding_mask.view(-1, utterance_padding_mask.size(-1))

        # to [utt_max_len, batch X dia_max_len, dim]
        dialogue = dialogue.view(-1, dialogue.size(2), dialogue.size(3)).transpose(0, 1)

        ### not using memory mask and prediction mask, using padding words mask
        utt_memory = self.utterance_encoder(dialogue, mask=None, src_key_padding_mask=utterance_padding_mask)
        utt_memory = utt_memory.transpose(0, 1)
        #utt_memory = torch.masked_fill(utt_memory, utterance_padding_mask.unsqueeze(2), 0)

        # using first col word embedding to represent the sentence
        utt_memory = utt_memory[:, 0, :].squeeze(1)

        # -> [batch, dia_max_len, dim]
        utt_memory = utt_memory.contiguous().view(dia_batch_size, -1, utt_memory.size(1))

        # add role embedding
        utt_memory = self.role_embedding(utt_memory, role)

        # ****************** Structural Context Encoder module **********************
        adj = inputs['dialogue_gat_attention_mask']
        # -> [batch, dia_max_len, dim]
        dia_memory = self.gat_encoder(utt_memory, adj)
        return dia_memory, utt_memory

    def decode_utterance(self, dia_memory, inputs):

        tgt_in = inputs['dialogue'][:, :, :-1]

        # token & position embedding
        tgt_in = self.embedding(tgt_in) * math.sqrt(self.d_model)
        tgt_in = self.position_embedding(tgt_in)

        tgt_in = tgt_in.view(-1, tgt_in.size(2), tgt_in.size(3))
        tgt_in = tgt_in.transpose(0, 1)

        ######################### GateTransformer Decoder #################################
        utterance_padding_mask = inputs['utterance_padding_mask'][:, :, 1:]
        utterance_padding_mask = utterance_padding_mask.view(-1, utterance_padding_mask.size(-1))
        # decoder attention mask
        utterance_decoder_attention_mask = inputs['utterance_decoder_attention_mask']
        # memory attention mask
        memory_attention_mask = inputs['memory_attention_mask']
        # memory padding mask
        memory_key_padding_mask = utterance_padding_mask

        # to [1, batchxdia_max_len, dim]
        dia_memory = dia_memory.contiguous().view(-1, dia_memory.size(2)).unsqueeze(0)

        # decode utterance
        utt_decoder_output = self.utterance_decoder(tgt_in, dia_memory, tgt_mask=utterance_decoder_attention_mask, memory_mask=memory_attention_mask, \
                                        tgt_key_padding_mask=utterance_padding_mask, memory_key_padding_mask=memory_key_padding_mask)

        utt_decoder_output = utt_decoder_output.transpose(0, 1)
        utt_decoder_output = self.fc(utt_decoder_output)

        return utt_decoder_output, utterance_padding_mask

    def calculate_loss(self, utt_decoder_outputs, tgt_out, decoder_utt_padding_mask, decoder_dia_padding_mask):
        utt_decoder_outputs = torch.masked_fill(utt_decoder_outputs, decoder_utt_padding_mask.unsqueeze(2), float('-inf'))

        utt_decoder_outputs = utt_decoder_outputs.transpose(1, 2)
        output = F.cross_entropy(utt_decoder_outputs, tgt_out, reduction='none')

        output = torch.masked_fill(output, decoder_utt_padding_mask, 0)

        loss = output.sum(dim=1) / torch.sum((decoder_utt_padding_mask == False), dim=1).float()

        # mask dialogs
        loss = torch.masked_fill(loss, decoder_dia_padding_mask.squeeze(1), 0) if decoder_dia_padding_mask is not None else loss

        loss = loss.sum(dim=0) / torch.sum((decoder_dia_padding_mask == False), dim=0).float() if decoder_dia_padding_mask is not None else loss.mean()

        return loss

    def forward(self, inputs):

        # encode all dialogue using UE and SCE
        dia_memory, utt_memory = self.encode(inputs)

        # decode utterance by GateTransformerDecoder
        utterance_decoder_outputs, utterance_padding_mask = self.decode_utterance(dia_memory, inputs)

        # calculate all reconstruction loss
        dialogue_padding_mask = inputs['dialogue_padding_mask'].view(-1, 1)
        tgt_out = inputs['dialogue'][:, :, 1:]
        tgt_out = tgt_out.view(-1, tgt_out.size(2))

        loss = self.calculate_loss(utterance_decoder_outputs, tgt_out, utterance_padding_mask, dialogue_padding_mask)

        return loss

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
