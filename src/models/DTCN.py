import os, sys, copy, json

sys.path.append(os.path.abspath("../"))
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal

import numpy as np
import pandas as pd
import logging, math

from models.embedding import SinCosPositionalEncoding, RolePositionalEncoding, DialogPositionalEncoding
from models.relation_loss import UCRL

from scipy.optimize import linear_sum_assignment
import utils
import config
import metrics, clustering

from models.autoencoder import AutoEncoderWithTransformer
from models.statistician import Statistic
from models.info_nce import InfoNCE

DEBUG = True
Verbose = True


class DTCN(nn.Module):

    def __init__(self, model_config, base_model_path=None, dialogue_clusters_number=0, utterance_clusters_number=0, dialogue_labels_path=None, utterance_labels_path=None, loss_rates={'ct': 0.5, 'ae': 1, 'rs': 1}, device=torch.device('cpu')):

        super(DTCN, self).__init__()
        self.device = device
        self.d_model = model_config['d_model']
        self.nhead = model_config['nhead']
        self.dim_feedforward = model_config['dim_feedforward']
        self.dropout = model_config['dropout']
        self.activation = model_config['activation']
        self.dialogue_encoder_layer_number = model_config['dialogue_encoder_layer_number']
        self.discount = model_config['discount']
        self.loss_rates = loss_rates

        with open(dialogue_labels_path, "r") as fp:
            self.dialogue_labels = json.load(fp)

        with open(utterance_labels_path, "r") as fp:
            self.utterance_labels = json.load(fp)

        self.dialogue_clusters_number = dialogue_clusters_number
        self.utterance_clusters_number = utterance_clusters_number

        # dialog encoder
        dialogue_encoder_layer = nn.TransformerEncoderLayer(self.d_model, self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, activation=self.activation)
        dialogue_encoder_norm = nn.LayerNorm(self.d_model)
        self.dialogue_encoder = nn.TransformerEncoder(dialogue_encoder_layer, self.dialogue_encoder_layer_number, norm=dialogue_encoder_norm)

        self._reset_parameters()

        # fusion
        self.fusion = nn.Linear(self.d_model + self.d_model, self.d_model)
        # self.gate_layer = nn.Linear(self.d_model * 2, self.d_model)

        # position embedding
        self.pos_embedding = DialogPositionalEncoding(self.d_model, dropout=self.dropout, max_len=100, device=self.device)

        # utt center loss
        self.ucrl = UCRL(self.utterance_clusters_number, self.d_model, self.device).to(self.device)

        # self.info_nce = InfoNCE(temperature=0.1, reduction='mean')

        self.soft_max_layer = nn.Linear(self.d_model, self.dialogue_clusters_number)
        self.utt_soft_max_layer = nn.Linear(self.d_model, self.utterance_clusters_number).to(self.device)
        # nn.init.orthogonal_(self.soft_max_layer.weight)
        # nn.init.sparse_(self.soft_max_layer.weight, sparsity=0.1)

        self.norm = nn.LayerNorm(dialogue_clusters_number)

        #########
        # statistic for itr training
        n_samples = len(self.dialogue_labels)
        session_id_to_voc = {}
        for i, id in enumerate(list(sorted(list(self.dialogue_labels)))):
            session_id_to_voc[str(id)] = i
        self.statistic_dialogue = Statistic(n_samples, self.dialogue_clusters_number, discount=self.discount, device=device)
        self.dialogue_labels, match_acc_dialogue = self.update_dialogue_labels(self.dialogue_labels)

        # statistic utterances for itr training
        n_samples = len(self.utterance_labels)
        session_id_to_voc_utt = {}
        for i, id in enumerate(list(sorted(list(self.utterance_labels)))):
            session_id_to_voc_utt[id] = i
        self.statistic_utterance = Statistic(n_samples, self.utterance_clusters_number, discount=self.discount, device=device)
        self.utterance_labels, match_acc_utterance = self.update_utterance_labels(self.utterance_labels)

        ###########
        self.base_model_path = base_model_path
        # AE
        ae = torch.load(base_model_path, map_location=self.device)
        ae.device = self.device
        self.ae = ae

    def update_dialogue_labels(self, dialogue_labels):
        return self.statistic_dialogue.update_labels(dialogue_labels)

    def update_utterance_labels(self, utterance_labels):
        return self.statistic_utterance.update_labels(utterance_labels)

    def set_labels(self, path):
        with open(path, "r") as fp:
            self.dialogue_labels = json.load(fp)

    def set_utterance_labels(self, path):
        with open(path, "r") as fp:
            self.utterance_labels = json.load(fp)

    def assignment_labels(self, batch):
        bts, dim = batch.size()

        n_clusters = self.dialogue_clusters_number
        #  calculate an [bts*n_cls, dim]
        batch = batch.unsqueeze(1)
        batch = batch.repeat(1, n_clusters, 1)
        batch = batch.view(-1, dim)

        # => [n_cls, dim] => [n_cls*bts, dim]
        centers_n = self.center_loss.centers.unsqueeze(0)

        centers_n = centers_n.repeat(bts, 1, 1)

        centers_n = centers_n.view(-1, dim)

        dist = torch.pow(batch - centers_n, 2).sum(1)

        dist = dist.view(-1, n_clusters)

        labels = dist.argmin(1)

        return labels

    def assignment_labels_sf(self, batch):
        probs = F.softmax(batch, dim=1)
        labels = probs.argmax(1)

        return labels

    def encode(self, inputs):

        dialogue = inputs['dialogue']
        roles = inputs['role']
        session_ids = inputs['id']
        dialogue_lengths = inputs['dialogue_length']

        dialogue_padding_mask = inputs['dialogue_padding_mask']

        dia_batch_size, dia_max_len, utt_max_len = dialogue.size()

        gat_memory, utt_memory = self.ae.encode(inputs)
        utt_embedding = gat_memory

        # ************ labels ***********
        # utterance labels
        utterance_labels = []
        for k, idx in enumerate(session_ids):
            for i in range(dialogue_lengths[k]):
                uid = "{}_{}".format(idx, i)
                uid = int(self.utterance_labels[uid])
                utterance_labels.append(uid)
            for i in range(dialogue_lengths[k], dia_max_len):
                uid = "{}_{}".format(idx, dialogue_lengths[k] - 1)
                utterance_labels.append(int(self.utterance_labels[uid]))
        utterance_labels = torch.LongTensor(utterance_labels).to(self.device)

        # ************ dialogue encoder *************
        utt_emb = self.ucrl.get_centers(utterance_labels.view(-1).long()).view(dia_batch_size, dia_max_len, -1)

        # linear fusion
        # utt_emb = self.fusion(torch.cat([utt_emb, gat_memory], dim=-1))
        utt_emb = utt_emb + gat_memory

        # add role information
        # utt_emb = self.ae.role_embedding(utt_emb, roles)
        utt_emb = self.pos_embedding(utt_emb, torch.arange(0, dia_max_len).to(self.device))

        utt_emb = utt_emb.transpose(0, 1)
        dia_memory = self.dialogue_encoder(utt_emb, mask=None, src_key_padding_mask=dialogue_padding_mask)
        dia_memory = dia_memory[0, :, :].squeeze(0)

        dia_memory = self.soft_max_layer(dia_memory)
        dia_memory = self.norm(dia_memory)

        return dia_memory, utt_embedding

    def forward(self, inputs):
        # dias: [batch, dia_max_len, utt_max_len]
        # dia_lens: [batch]
        # utt_lens: [batch, dia_max_len]

        dialogue = inputs['dialogue']
        roles = inputs['role']
        session_ids = inputs['id']
        dialogue_lengths = inputs['dialogue_length']

        utterance_padding_mask = inputs['utterance_padding_mask']

        dia_batch_size, dia_max_len, utt_max_len = dialogue.size()

        gat_memory, utt_memory = self.ae.encode(inputs)

        # ***************  AE step  ****************
        # decode utterance by GateTransformerDecoder
        utterance_decoder_outputs, utterance_padding_mask = self.ae.decode_utterance(gat_memory, inputs)
        # calculate all reconstruction loss
        dialogue_padding_mask = inputs['dialogue_padding_mask']
        tgt_out = inputs['dialogue'][:, :, 1:]
        tgt_out = tgt_out.view(-1, tgt_out.size(2))

        # reconstruction loss
        ae_loss = self.ae.calculate_loss(utterance_decoder_outputs, tgt_out, utterance_padding_mask, dialogue_padding_mask.view(-1, 1))

        # ************ labels ***********
        # dialogue_labels
        dialogue_labels = []
        for id in session_ids:
            dialogue_labels.append(int(self.dialogue_labels[str(id)]))
        dialogue_labels = torch.LongTensor(dialogue_labels).to(self.device)

        # utt labels
        utterance_labels = []
        for k, idx in enumerate(session_ids):
            for i in range(dialogue_lengths[k]):
                uid = "{}_{}".format(idx, i)
                uid = int(self.utterance_labels[uid])
                utterance_labels.append(uid)
            for i in range(dialogue_lengths[k], dia_max_len):
                uid = "{}_{}".format(idx, dialogue_lengths[k] - 1)
                utterance_labels.append(int(self.utterance_labels[uid]))
        utterance_labels = torch.LongTensor(utterance_labels).to(self.device)

        # ************ ucrl loss ******************
        r_loss = self.ucrl(utt_memory, utterance_labels.view(utt_memory.size(0), -1), dialogue_padding_mask[:, 1:])

        # ************ dialogue encoder *************
        utt_emb = self.ucrl.get_centers(utterance_labels.view(-1).long()).view(dia_batch_size, dia_max_len, -1)
        # utt_emb = self.fusion(torch.cat([utt_emb, gat_memory], dim=-1))
        utt_emb = utt_emb + gat_memory

        # add role information
        # utt_emb = self.ae.role_embedding(utt_emb, roles)
        utt_emb = self.pos_embedding(utt_emb, torch.arange(0, dia_max_len).to(self.device))

        utt_emb = utt_emb.transpose(0, 1)
        dia_memory = self.dialogue_encoder(utt_emb, mask=None, src_key_padding_mask=dialogue_padding_mask)
        dia_memory = dia_memory[0, :, :].squeeze(0)

        # ************ info nce loss **************

        # ************ softmax loss ************
        out = self.soft_max_layer(dia_memory)
        out = self.norm(out)

        # info_nce_loss = self.info_nce(out, out)

        sf_loss = F.cross_entropy(out, dialogue_labels)

        # ************ all loss ************
        ae_w = self.loss_rates['ae']
        rs_w = self.loss_rates['rs']
        kl_w = self.loss_rates['kl']

        if Verbose:
            print("# ae_loss: {:.5f}, relation_loss: {:.5f}, sf: {:.5f}, infonce:  {:.5f}".format(ae_loss.item(), r_loss.item(), sf_loss.item(), 0))

        loss = rs_w * r_loss + ae_w * ae_loss + kl_w * sf_loss
        return loss

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
