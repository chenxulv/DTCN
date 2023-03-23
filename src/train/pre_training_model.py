import os, sys, time, copy, random, json

sys.path.append(os.path.abspath("../"))
sys.path.append(os.path.abspath("../../"))

import matplotlib as mpl

mpl.use('agg')
import matplotlib.pyplot as plt

import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from tqdm import tqdm
import argparse

from models.autoencoder import AutoEncoderWithTransformer
from models.optim import ScheduledOptim_V3
from dataset import DialogDatasetOriginal, Collate, load_data
import utils, config, clustering, metrics


def train(args):
    # init device
    if args.gpu < 0 or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")

    dataset_name = config.MODEL_CONFIG['dataset_name']
    params = config.MODEL_CONFIG['pre_training']

    train_dir = os.path.join(config.DATA_DIR, dataset_name, 'train')
    data, voc = load_data(train_dir, 'train.json', "voc.json")
    train_data = DialogDatasetOriginal(data, voc)
    train_loader = DataLoader(train_data, batch_size=params["train_batch_size"], shuffle=True, drop_last=False, num_workers=3, collate_fn=Collate(voc=voc))
    print("data ready...")

    utils.init_dirs(dataset_name, params['store_id'])

    init_lr = params["init_lr"]
    #warm_up_steps = int(len(train_loader) * params['train_epoch'] * 0.1) # params["warm_up_steps"]

    warm_up_steps = params["warm_up_steps"]
    print("warm up steps: ", warm_up_steps)

    ae = AutoEncoderWithTransformer(train_data.voc, params, device=device).to(device)

    # create a optimizer
    optimizer = ScheduledOptim_V3(optim.Adam(ae.parameters(), betas=(0.9, 0.98), eps=1e-09), init_lr, params['d_model'], warm_up_steps)

    # summary all model information
    print("# voc size:       {}".format(len(train_data.voc)))
    param_num = sum(param.numel() for param in ae.parameters())
    print('# net parameters: {} or {:.2f}M'.format(param_num, param_num / 1024 / 1024))
    print('#', ae)

    # train model
    ae.train()
    for ep in tqdm(range(1, params["train_epoch"] + 1)):
        print("#{}".format(ep))
        total_loss, total_ppl = 0, 0
        for i, batch in enumerate(tqdm(train_loader)):
            inputs = batch
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(device)

            if i % 200 != 0 or i == 0:
                optimizer.zero_grad()
                loss = ae(inputs)
                loss.backward()
                optimizer.step_and_update_lr()

                ppl = np.exp(loss.item())
                total_loss += loss.item()
                total_ppl += ppl

                print("train loss: {:.5f}, ppl: {:.5f}".format(loss.item(), ppl))
            else:
                ae.eval()
                with torch.no_grad():
                    loss = ae(inputs)
                    # calculate perplexity
                    ppl = np.exp(loss.item())
                    total_loss += loss.item()
                    total_ppl += ppl
                    print("val loss: {:.5f}, ppl: {:.5f}".format(loss.item(), ppl))
                ae.train()

        if ep % 10 == 0:
            file_path = os.path.join(config.RESULT_DIR, dataset_name, params["store_id"], 'models', "base_encoder_epoch_{}.pth".format(ep))
            torch.save(ae, file_path)


def get_embedding(args, epoch=100):
    utils.seed_torch(seed=42)

    # init device
    if args.gpu < 0 or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")

    dataset_name = config.MODEL_CONFIG['dataset_name']
    params = config.MODEL_CONFIG['pre_training']

    train_dir = os.path.join(config.DATA_DIR, dataset_name, 'train')
    data, voc = load_data(train_dir, 'train.json', "voc.json")
    train_data = DialogDatasetOriginal(data, voc)
    train_loader = DataLoader(train_data, batch_size=params["train_batch_size"], shuffle=False, drop_last=False, num_workers=2, collate_fn=Collate(voc=voc))

    print("data ready...")
    file_path = os.path.join(config.RESULT_DIR, dataset_name, str(params["store_id"]), 'models', "base_encoder_epoch_{}.pth".format(epoch))

    ae = torch.load(file_path, map_location=device)
    ae.device = device
    ae = ae.to(device)

    utils.init_dirs(dataset_name, params["store_id"])

    # eval model
    ae.eval()
    all_dia_embedding = None
    all_utt_embedding = None
    for i, batch in enumerate(tqdm(train_loader)):
        inputs = batch
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(device)

        session_ids = inputs['id']

        # output embeddings
        dia_memory, utt_memory = ae.encode(inputs)
        utt_embedding = dia_memory

        # dialogue padding mask
        dialogue_padding_mask = inputs['dialogue_padding_mask']

        # calcuate init dialogue embedding
        valid_sum = torch.sum((dialogue_padding_mask == False), dim=1).view(-1, 1).float()
        dia_memory = dia_memory.masked_fill(dialogue_padding_mask.unsqueeze(2), 0)
        dia_memory = dia_memory.sum(1)
        dia_memory = dia_memory / valid_sum
        #dia_memory = torch.max(dia_memory, 1)[0]
        dia_memory = dia_memory.view(-1, dia_memory.size(-1))

        # record all utterance and dialogue embeddings
        all_utt_embedding = pd.concat([all_utt_embedding, pd.DataFrame({'session_id': session_ids, 'embedding': utt_embedding.detach().cpu().numpy().tolist(), 'dia_lens': inputs['dialogue_length'].detach().cpu().numpy().tolist()})])
        all_dia_embedding = pd.concat([all_dia_embedding, pd.DataFrame({'session_id': session_ids, 'embedding': dia_memory.detach().cpu().numpy().tolist()})])

    file_path = os.path.join(config.RESULT_DIR, dataset_name, params["store_id"], 'embeddings', "utt_init_embedding_{}.pkl".format(epoch))
    all_utt_embedding.to_pickle(file_path)

    file_path = os.path.join(config.RESULT_DIR, dataset_name, params["store_id"], 'embeddings', "dialog_init_embedding_{}.pkl".format(epoch))
    all_dia_embedding.to_pickle(file_path)

    return all_dia_embedding, all_utt_embedding


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', default=1, type=int)

    args = parser.parse_args()

    params = config.MODEL_CONFIG["pre_training"]
    print(json.dumps(params, ensure_ascii=False, indent=4))

    train(args)

    get_embedding(args, epoch=params['train_epoch'])
