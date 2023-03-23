import os, sys, time, copy, random

sys.path.append(os.path.abspath("../"))

import matplotlib as mpl

mpl.use('agg')

import matplotlib.pyplot as plt

import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import json
from sklearn.utils import shuffle
import argparse

from models.autoencoder import AutoEncoderWithTransformer
from models.optim import ScheduledOptim, ScheduledOptim_V2
from models.DTCN import DTCN
from models.statistician import Statistic
from dataset import Collate, load_data, DialogDatasetOriginal
import utils, config

import clustering

#from metric.acc_dia import test_acc_tsne, test_acc_tsne_utt
import metrics

from tqdm import tqdm
import argparse

SEED = None


def get_embedding(dcae, loader, epoch, device, args):
    all_utt_embedding = None
    all_dia_embedding = None
    all_dia_labels = None
    all_dia_labels_sf = None

    params = config.MODEL_CONFIG['joint_training']

    dataset_name = config.MODEL_CONFIG['dataset_name']
    store_id = params["store_id"]

    for i, batch in enumerate(tqdm(loader)):
        inputs = batch
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(device)

        session_ids = inputs['id']
        dialogue_lengths = inputs['dialogue_length']

        with torch.no_grad():
            dia_memory, utt_embedding = dcae.encode(inputs)
        dia_memory = dia_memory.view(-1, dia_memory.size(-1))

        all_utt_embedding = pd.concat([all_utt_embedding, pd.DataFrame({'session_id': session_ids, 'embedding': utt_embedding.detach().cpu().numpy().tolist(), 'dia_lens': dialogue_lengths.detach().cpu().numpy().tolist()})])

        all_dia_embedding = pd.concat([all_dia_embedding, pd.DataFrame({'session_id': session_ids, 'embedding': dia_memory.detach().cpu().numpy().tolist()})])

        labels = dcae.assignment_labels_sf(dia_memory).cpu().numpy().tolist()
        all_dia_labels_sf = pd.concat([all_dia_labels_sf, pd.DataFrame({'session_id': session_ids, 'labels': labels})])

    file_path = os.path.join(config.RESULT_DIR, dataset_name, store_id, 'embeddings', "utterance_epoch_{}_embedding.pkl".format(epoch))
    all_utt_embedding.to_pickle(file_path)

    file_path = os.path.join(config.RESULT_DIR, dataset_name, store_id, 'embeddings', "dialog_epoch_{}_embedding.pkl".format(epoch))
    all_dia_embedding.to_pickle(file_path)

    labels = dict(zip(all_dia_labels_sf['session_id'].values.tolist(), all_dia_labels_sf['labels'].values.tolist()))

    file_path = os.path.join(config.RESULT_DIR, dataset_name, store_id, 'labels', 'labels_{}_dialog_sf.json'.format(epoch))
    with open(file_path, 'w') as fp:
        json.dump(labels, fp, indent=4)

    return all_dia_embedding, all_utt_embedding


def train(args):
    # init device
    if args.gpu < 0 or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")

    params = config.MODEL_CONFIG['joint_training']

    dataset_name = config.MODEL_CONFIG['dataset_name']
    store_id = params["store_id"]

    dialogue_clusters_number = config.MODEL_CONFIG["clustering"]['dataset'][dataset_name]['dialogue']
    utterance_clusters_number = config.MODEL_CONFIG["clustering"]['dataset'][dataset_name]['utterance']
    dialogue_clustering_method = config.MODEL_CONFIG["clustering"]['method']['dialogue']
    utterance_clustering_method = config.MODEL_CONFIG["clustering"]['method']['utterance']

    loss_rates = params['loss_rates']
    base_model_file_name = params['base_model_file_name']
    init_dialog_labels_file_name = params['init_dialog_labels_file_name']
    init_utterance_labels_file_name = params['init_utterance_labels_file_name']

    train_batch_size = params['train_batch_size']
    init_lr = params['init_lr']
    max_lr = params['max_lr']
    d_model = params['d_model']

    dialogue_update_interval = params['dialogue_update_interval']
    joint_training_epoch = params['train_epoch']

    # load data
    train_dir = os.path.join(config.DATA_DIR, dataset_name, 'train')
    data, voc = load_data(train_dir, 'train.json', "voc.json")
    train_data = DialogDatasetOriginal(data, voc)
    test_data = DialogDatasetOriginal(data, voc)
    utils.init_dirs(dataset_name, store_id)
    print("data ready...")
    # create a model

    base_model_path = os.path.join(config.RESULT_DIR, dataset_name, store_id, 'models', base_model_file_name)
    init_dialog_labels_path = os.path.join(config.RESULT_DIR, dataset_name, store_id, 'labels', init_dialog_labels_file_name)
    init_utterance_labels_path = os.path.join(config.RESULT_DIR, dataset_name, store_id, 'labels', init_utterance_labels_file_name)

    dcae = DTCN(config.MODEL_CONFIG['joint_training'],
                dialogue_clusters_number=dialogue_clusters_number,
                utterance_clusters_number=utterance_clusters_number,
                base_model_path=base_model_path,
                dialogue_labels_path=init_dialog_labels_path,
                utterance_labels_path=init_utterance_labels_path,
                loss_rates=loss_rates,
                device=device)

    dcae = dcae.to(device)
    dcae.device = device

    # create a optimizer
    optimizer = None
    # optimizer = ScheduledOptim(optim.Adam(dcae.parameters(), betas=(0.9, 0.98), eps=1e-09), init_lr, d_model, warm_up_steps, max_lr=args.max_lr)
    # optimizer = ScheduledOptim_V2(optim.Adam(dcae.parameters(), betas=(0.9, 0.98), eps=1e-09), 1e-6, 1e-3, warm_up_steps)

    # summary all model information
    print("# voc size:       {}".format(len(train_data.voc)))
    param_num = sum(param.numel() for param in dcae.parameters())
    print('# net parameters: {} or {:.2f}M'.format(param_num, param_num / 1024 / 1024))
    print('#', dcae)

    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, drop_last=True, num_workers=1, collate_fn=Collate(voc=train_data.voc))
    optimizer = ScheduledOptim(optim.Adam(dcae.parameters(), betas=(0.9, 0.98), eps=1e-09), init_lr, d_model, len(train_loader) * dialogue_update_interval, max_lr=max_lr)
    # optimizer = optim.Adam(dcae.parameters(), lr=5e-4, betas=(0.9, 0.98), eps=1e-09)

    dcae.train()
    for ep in tqdm(range(1, joint_training_epoch + 1)):
        print("#epoch {}".format(ep))
        # train one epoch
        for i, batch in enumerate(tqdm(train_loader)):
            inputs = batch
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(device)

            optimizer.zero_grad()
            loss = dcae(inputs)
            loss.backward()
            optimizer.step_and_update_lr()
            # optimizer.step()
            print("train loss: {:.5f}, lr: {:.5f}".format(loss.item(), optimizer.lr))

        # save model
        file_path = os.path.join(config.RESULT_DIR, dataset_name, store_id, 'models', 'model_epoch_{}.pth'.format(ep))
        torch.save(dcae, file_path)

        # ############# clustering
        loader = DataLoader(test_data, batch_size=train_batch_size, shuffle=False, drop_last=False, num_workers=1, collate_fn=Collate(voc=test_data.voc))
        # eval model
        dcae.eval()
        all_dia_embedding, all_utt_embedding = get_embedding(dcae, loader, ep, device, args)
        dcae.train()

        dialog_labels_path = os.path.join(config.RESULT_DIR, dataset_name, store_id, 'labels', 'labels_{}_dialogue.json'.format(ep))
        dialogue_labels = clustering.clustering(
            type='dialogue',
            df=all_dia_embedding,
            cluster_num=dcae.dialogue_clusters_number,
            method=dialogue_clustering_method,
            #init_means=dcae.proto_loss.centers.detach().cpu().numpy(),
            init_means=None,
            save_path=dialog_labels_path,
            seed=SEED)
        del all_dia_embedding
        true_labels = clustering.load_true_labels(dataset_name)
        metrics.metric_from_ids(true_labels, dialogue_labels)

        print(flush=True)
        # update dia clusters
        if ep != 0 and ep % params['dialogue_update_interval'] == 0:
            print("#update dialog: {}".format(ep // params['dialogue_update_interval']))
            print("# assignment labels by count matrix", end='')
            dcae.dialogue_labels, match_acc_dialogue = dcae.update_dialogue_labels(dialogue_labels)
            true_labels = clustering.load_true_labels(dataset_name)
            metrics.metric_from_ids(true_labels, dcae.dialogue_labels)
        else:
            print("# assignment labels by count matrix", end='')
            dia_labels, match_acc_dialogue = dcae.update_dialogue_labels(dialogue_labels)
            true_labels = clustering.load_true_labels(dataset_name)
            metrics.metric_from_ids(true_labels, dia_labels)

        file_path = os.path.join(config.RESULT_DIR, dataset_name, store_id, 'labels', 'labels_{}_dialog_sf.json'.format(ep))
        with open(file_path, 'r') as fp:
            y = json.load(fp)
        print("# assignment labels by softmax", end=' ')
        metrics.metric_from_ids(true_labels, y)

        # clustering utterance
        utterance_labels_path = os.path.join(config.RESULT_DIR, dataset_name, store_id, 'labels', 'labels_{}_utterance.json'.format(ep))
        utt_labels = clustering.clustering(type='utterance', df=all_utt_embedding, cluster_num=dcae.utterance_clusters_number, method=utterance_clustering_method, save_path=utterance_labels_path, seed=SEED)
        print(flush=True)

        del all_utt_embedding

        # update utt clusters
        if ep != 0 and ep % params['utterance_update_interval'] == 0:
            print("#update utterance: {}".format(ep // params['utterance_update_interval']))
            dcae.utterance_labels, match_acc_utterance = dcae.update_utterance_labels(utt_labels)
        else:
            _, match_acc_utterance = dcae.update_utterance_labels(utt_labels)

        print('==> match pre-iteration acc (utterance): {:.5f}'.format(match_acc_utterance))
        print('==> match pre-iteration acc (dialogue) : {:.5f}'.format(match_acc_dialogue))

        plt.close()
        centers_path_utt = os.path.join(config.RESULT_DIR, dataset_name, store_id, 'centers', 'centers_{}_utt.pkl'.format(ep))
        pd.to_pickle(dcae.ucrl.centers.detach().cpu().numpy(), centers_path_utt)

        if match_acc_dialogue >= 0.999:
            print('finish ....')
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', default=3, type=int)
    args = parser.parse_args()

    params = config.MODEL_CONFIG
    print(json.dumps(params, ensure_ascii=False, indent=4))

    train(args)
