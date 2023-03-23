import os, sys, time, copy, random, json

sys.path.append(os.path.abspath("../"))
sys.path.append(os.path.abspath("../../"))

import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import utils, config, clustering, metrics
from models.statistician import Statistic


def init_clustering_evaluation(pred_path, true_path):
    # id_to_true
    with open(true_path, 'r') as fp:
        id_to_true = json.load(fp)

    # id_to_pred
    with open(pred_path, 'r') as fp:
        id_to_pred = json.load(fp)

    y_pred, y_true = [], []
    for k, v in id_to_pred.items():
        y_pred.append(v)
        y_true.append(id_to_true[k])

    # evaluation
    metrics.metric(y_true, y_pred)


def init_clustering(args):

    params = config.MODEL_CONFIG['pre_training']

    dataset_name = config.MODEL_CONFIG['dataset_name']
    store_id = params["store_id"]

    for i in range(args.times):
        # clustering dialogues
        file_path = os.path.join(config.RESULT_DIR, dataset_name, store_id, 'embeddings', "dialog_init_embedding_{}.pkl".format(args.epoch))

        save_path = os.path.join(config.RESULT_DIR, dataset_name, store_id, 'labels', "labels_init_{}_dialogue_{}.json".format(args.epoch, i))

        df = pd.read_pickle(file_path)

        cluster_num = config.MODEL_CONFIG['clustering']['dataset'][dataset_name]['dialogue']
        print("dialogue clusters: {}".format(cluster_num))
        clustering.clustering(type='dialogue', df=df, cluster_num=config.MODEL_CONFIG['clustering']['dataset'][dataset_name]['dialogue'], method=config.MODEL_CONFIG['clustering']['method']['dialogue'], save_path=save_path)

        # evalution dialogue clusters
        true_path = os.path.join(config.DATA_DIR, dataset_name, 'train', 'session_to_task.json')

        init_clustering_evaluation(save_path, true_path)

        # clustering utterances
        file_path = os.path.join(config.RESULT_DIR, dataset_name, store_id, 'embeddings', "utt_init_embedding_{}.pkl".format(args.epoch))

        save_path = os.path.join(config.RESULT_DIR, dataset_name, store_id, 'labels', "labels_init_{}_utterance_{}.json".format(args.epoch, i))

        df = pd.read_pickle(file_path)
        cluster_num = config.MODEL_CONFIG['clustering']['dataset'][dataset_name]['utterance']
        print("utterance clusters: {}".format(cluster_num))
        clustering.clustering(type='utterance', df=df, cluster_num=config.MODEL_CONFIG['clustering']['dataset'][dataset_name]['utterance'], method=config.MODEL_CONFIG['clustering']['method']['utterance'], save_path=save_path)

        print("", end='', flush=True)


def init_assignments(args, type="dialogue"):

    pre_params = config.MODEL_CONFIG['pre_training']

    dataset_name = config.MODEL_CONFIG['dataset_name']

    idx = 0

    if type == "dialogue":
        file_path = os.path.join(config.RESULT_DIR, dataset_name, pre_params["store_id"], 'labels', "labels_init_{}_dialogue_{}.json".format(args.epoch, idx))
        n_clusters = config.MODEL_CONFIG["clustering"]['dataset'][dataset_name]['dialogue']
    else:
        file_path = os.path.join(config.RESULT_DIR, dataset_name, pre_params["store_id"], 'labels', "labels_init_{}_utterance_{}.json".format(args.epoch, 0))
        n_clusters = config.MODEL_CONFIG["clustering"]['dataset'][dataset_name]['utterance']

    with open(file_path, 'r') as fp:
        datas = json.load(fp)

    n_samples = len(datas)
    print("n_clusters", n_clusters)
    print('n_samples', n_samples)

    stat = Statistic(n_samples, n_clusters, discount=1)

    idxes = list(range(args.times))
    random.shuffle(idxes)
    for i in idxes:
        if type == "dialogue":
            file_path = os.path.join(config.RESULT_DIR, dataset_name, pre_params["store_id"], 'labels', "labels_init_{}_dialogue_{}.json".format(args.epoch, i))
            # evalution dialogue clusters
            true_path = os.path.join(config.DATA_DIR, dataset_name, 'train', 'session_to_task.json')
            init_clustering_evaluation(file_path, true_path)
        else:
            file_path = os.path.join(config.RESULT_DIR, dataset_name, pre_params["store_id"], 'labels', "labels_init_{}_utterance_{}.json".format(args.epoch, i))

        # update
        with open(file_path, 'r') as fp:
            labels = json.load(fp)

        new_labels, acc = stat.update_labels(labels)
        print('match rate of assignment between previous and current iteration: {:.5f}'.format(acc))

    if type == "dialogue":
        save_path = os.path.join(config.RESULT_DIR, dataset_name, pre_params["store_id"], 'labels', "labels_{}_dialogue.json".format(0))
    else:
        save_path = os.path.join(config.RESULT_DIR, dataset_name, pre_params["store_id"], 'labels', "labels_{}_utterance.json".format(0))

    with open(save_path, 'w') as fp:
        json.dump(new_labels, fp, indent=4)

    if type == "dialogue":
        # evalution dialogue clusters
        true_path = os.path.join(config.DATA_DIR, dataset_name, 'train', 'session_to_task.json')
        init_clustering_evaluation(save_path, true_path)

    # initial utterance assignments
    file_path = os.path.join(config.RESULT_DIR, dataset_name, pre_params["store_id"], 'labels', "labels_init_{}_utterance_{}.json".format(args.epoch, 0))
    taget_path = os.path.join(config.RESULT_DIR, dataset_name, pre_params["store_id"], 'labels', "labels_{}_utterance.json".format(0))

    os.popen("cp {} {}".format(file_path, taget_path))

    return new_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--times', default=10, type=int)
    args = parser.parse_args()

    init_clustering(args)

    init_assignments(args, type="dialogue")
