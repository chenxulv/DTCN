import os, sys, time, copy, random, json, argparse
sys.path.append(os.path.abspath("../"))

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from torch.utils.data import Dataset, DataLoader
import torch
import utils, config


class DialogDatasetOriginal(Dataset):
    def __init__(self, data: dict, voc: dict):
        self.voc = voc
        self.data = data
        self.session_ids = self.data['session_id']
        self.dialogues = self.data['utterances']
        self.utterance_lengths = self.data['utterance_length']
        self.dialogue_lengths = self.data['dialogue_length']
        self.roles = self.data['roles']

    def __getitem__(self, index):
        dias = copy.deepcopy(self.dialogues[index])
        roles = copy.deepcopy(self.roles[index])
        dialogue_lengths = copy.deepcopy(self.dialogue_lengths[index])
        utterance_lengths = copy.deepcopy(self.utterance_lengths[index])
        session_id = self.session_ids[index]

        return dias, roles, dialogue_lengths, utterance_lengths, session_id

    def __len__(self):
        return len(self.data['utterances'])


class Collate(object):
    def __init__(self, voc={}):
        self.voc = voc

    def padding_dialogue(self, dialogues, utterance_lengths, dialogue_lengths, roles):
        max_dialogue_length = dialogue_lengths.max()

        for i, dia in enumerate(dialogues):
            if len(dia) < max_dialogue_length:
                for _ in range(max_dialogue_length-len(dia)):
                    dialogues[i].append([self.voc['[start_utt]'], self.voc['[end_utt]']])
                    utterance_lengths[i].append(2)
                    roles[i].append(0)

        return dialogues, utterance_lengths, dialogue_lengths, roles

    def padding_utterance(self, dialogues, utterance_lengths):
        max_utterance_length = utterance_lengths.max()
        for i, dia in enumerate(dialogues):
            for j, utt in enumerate(dia):
                if len(utt) < max_utterance_length:
                    dialogues[i][j] = utt + [self.voc['[pad]']] * (max_utterance_length-len(utt))
        return dialogues

    def collate_func(self, batch):
        dialogues, roles, dialogue_lengths, utterance_lengths, session_id = [], [], [], [], []
        for bt in batch:
            bt = list(bt)
            dialogues.append(bt[0])
            roles.append(bt[1])
            dialogue_lengths.append(bt[2])
            utterance_lengths.append(bt[3])
            session_id.append(bt[4])

        dialogue_lengths = np.array(dialogue_lengths)
        dialogues, utterance_lengths, dialogue_lengths, roles = self.padding_dialogue(dialogues, utterance_lengths, dialogue_lengths, roles)

        utterance_lengths = np.array(utterance_lengths)
        dialogues = self.padding_utterance(dialogues, utterance_lengths)

        # to tensor
        dialogues = torch.LongTensor(dialogues)
        utterance_lengths = torch.LongTensor(utterance_lengths)
        dialogue_lengths = torch.LongTensor(dialogue_lengths)
        roles = torch.LongTensor(roles)

        max_dialogue_length, max_utterance_length = dialogue_lengths.max(), utterance_lengths.max()

        # ************************* masks *************************
        # padding mask for utterance encoder
        utterance_padding_mask = utils.generate_padding_mask(utterance_lengths.view(-1, 1), max_utterance_length).view(utterance_lengths.size(0), utterance_lengths.size(1), -1)
        # padding mask for dialogue encoder
        dialogue_padding_mask = utils.generate_padding_mask(dialogue_lengths.view(-1, 1), max_dialogue_length)
        dialogue_padding_mask[:, 0] = True
        # attention mask for GAT
        dialogue_gat_attention_mask = utils.generate_window_mask_matrix(max_dialogue_length, max_dialogue_length, window_size=config.MODEL_CONFIG['pre_training']['window_size'])
        adj = dialogue_gat_attention_mask.float().masked_fill(dialogue_gat_attention_mask, 0).masked_fill(dialogue_gat_attention_mask == False, 1)
        adj = adj.unsqueeze(0).repeat(dialogue_lengths.size(0), 1, 1)
        for i in range(adj.size(0)):
            adj[i][dialogue_lengths[i].item():, :] = 0
            adj[i][:, dialogue_lengths[i].item():] = 0
        # attention mask for utterance decoder
        utterance_decoder_attention_mask = utils.generate_square_subsequent_mask(max_utterance_length-1)
        # memory attention mask for utterance decoder
        memory_attention_mask = utils.generate_rectangle_subsequent_mask(max_utterance_length-1, max_utterance_length-1)

        inputs = {'dialogue': dialogues, 'role': roles, 'id': session_id,
                    'utterance_length': utterance_lengths, 'dialogue_length': dialogue_lengths,
                    'utterance_padding_mask': utterance_padding_mask, 'dialogue_padding_mask': dialogue_padding_mask,
                    'utterance_decoder_attention_mask': utterance_decoder_attention_mask, 'dialogue_gat_attention_mask': adj,
                    'memory_attention_mask': memory_attention_mask}

        return inputs

    def __call__(self, batch):
        return self.collate_func(batch)


def load_data(dir_path, data_file_name, voc_file_name):

    voc_path = os.path.join(dir_path, voc_file_name)
    data_file_path = os.path.join(dir_path, data_file_name)

    if not os.path.exists(voc_path) or not os.path.exists(data_file_path):
        raise Exception("not exist voc path: {}\nor data file: {}".format(voc_path, data_file_path))

    print('voc path', voc_path)
    with open(voc_path, 'r') as fp:
        voc = json.load(fp)

    with open(data_file_path, 'r') as fp:
        data = json.load(fp)
    return data, voc
