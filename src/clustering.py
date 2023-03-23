import os, sys, time, json, random

sys.path.append(os.path.abspath("../"))

from collections import defaultdict, Counter
import copy
import pandas as pd
import numpy as np

import matplotlib as mpl

mpl.use('agg')
import matplotlib.pyplot as plt

import seaborn

from sklearn.manifold import TSNE
from scipy.stats import t
from scipy.stats import norm, multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import argparse

# sklearn
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
import seaborn as sns
import config
import metrics
from metrics import calculate_acc, calculate_purity, calculate_ari, calculate_nmi


class MiniBatchGMM(object):

    def __init__(self, n_components, n_examples=10000, seed=None):
        self.model = GaussianMixture(n_components=n_components, covariance_type='tied', init_params='k-means++', random_state=seed)
        self.n_examples = n_examples

    def fit(self, X):

        ind = np.random.choice(X.shape[0], self.n_examples, replace=False)
        X = X[ind, :]

        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)


def clustering_dialog(embeddings, n_clusters=0, method='kmeans', init_means=None, seed=None):
    X = embeddings
    if method == 'kmeans':
        if init_means is not None:
            model = KMeans(n_clusters=n_clusters, init=init_means, random_state=seed)
        else:
            model = KMeans(n_clusters=n_clusters, random_state=seed)
        # model = MiniBatchKMeans(n_clusters=n_clusters, batch_size=10000)#, random_state=42)
        y_pred = model.fit_predict(X)
        centers = model.cluster_centers_

    elif method == 'gmm':
        if init_means is not None:
            model = GaussianMixture(n_components=n_clusters, covariance_type='tied', means_init=init_means, random_state=seed)
        else:
            model = GaussianMixture(n_components=n_clusters, covariance_type='tied', random_state=seed)  #, max_iter=200, tol=5e-4)
        y_pred = model.fit_predict(X)

    elif method == 'agg':
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="single", affinity="euclidean")
        y_pred = model.fit_predict(X)

    del model
    return y_pred


def clustering_utterance(embeddings, n_clusters=0, method='kmeans', seed=None):
    X = embeddings
    if method == 'kmeans':
        model = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1024, random_state=seed, init='k-means++', n_init='auto')  #, random_state=42)
        #model = KMeans(n_clusters=n_clusters)
        y_pred = model.fit_predict(X)

    elif method == 'gmm':
        #model = GaussianMixture(n_components=n_clusters, covariance_type='tied', random_state=42)
        model = MiniBatchGMM(n_components=n_clusters, n_examples=10000, seed=seed)
        # predict labels and centers
        model.fit(X)
        y_pred = model.predict(X)

    elif method == 'agg':
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward", affinity="euclidean")
        y_pred = model.fit_predict(X)

    return y_pred


def load_true_labels(dataset_name):
    id_to_true_labels_path = os.path.join(config.DATA_DIR, dataset_name, 'train', 'session_to_task.json')
    # get id_to_true_labels
    with open(id_to_true_labels_path, 'r') as fp:
        session_id_to_task = json.load(fp)
        session_id_to_true_label = session_id_to_task

    return session_id_to_true_label


def clustering(type='dialogue', df=None, cluster_num=0, method='gmm', init_means=None, save_path='', seed=None):
    if type == 'utterance':
        # clustering data
        X = np.array(df['embedding'].values.tolist(), dtype=object)
        ids = df['session_id'].values.tolist()
        dia_lens = df['dia_lens'].values.tolist()
        tmp, utt_labels = [], []
        for k, v in enumerate(X):
            tmp.append(np.array(v[:dia_lens[k]]))
            for i in range(dia_lens[k]):
                utt_labels.append("{}_{}".format(ids[k], i))
        X = np.concatenate(tmp)

        # clustering
        y_pred = clustering_utterance(X, cluster_num, method=method, seed=seed)

        # id_to_predict_labels
        id_to_predict_labels = {}
        for k, v in enumerate(utt_labels):
            id_to_predict_labels[v] = str(y_pred[k])

    elif type == 'dialogue':
        # clustering data
        X = np.array(df['embedding'].values.tolist())
        ids = df['session_id'].values.tolist()

        # clustering
        y_pred = clustering_dialog(X, cluster_num, method=method, init_means=init_means, seed=seed)

        # {session id: y_pred}
        id_to_predict_labels = {}
        for k, v in enumerate(ids):
            id_to_predict_labels[v] = str(y_pred[k])

    if save_path != '':
        with open(save_path, 'w') as fp:
            json.dump(id_to_predict_labels, fp, indent=4)

    return id_to_predict_labels


if __name__ == '__main__':
    pass
