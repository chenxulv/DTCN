
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
    def __init__(self, n_components, n_examples=5000):
        self.model = GaussianMixture(n_components=n_components, covariance_type='tied')
        self.n_examples = n_examples

    def fit(self, X):
        np.random.shuffle(X)
        X = X[:self.n_examples]
        self.model.fit(X)
        return self.model

    def predict(self, X):
        return self.model.predict(X)

def select_by_BIC(X, n_clusters=[10, 20, 30, 40, 50, 60, 70, 80, 100], random_state=42):
    np.random.shuffle(X)
    X = X[:50000]

    print('fitting... ...')
    models = [MiniBatchGMM(n).fit(X) for n in n_clusters]
    print('bicing...')
    bics = [m.bic(X) for m in models]
    print('aicing...')
    aics = [m.aic(X) for m in models]

    print(bics)

    print(aics)

    plt.plot(n_clusters, bics, label='BIC')
    plt.plot(n_clusters, aics, label='AIC')
    plt.legend(loc='best')
    plt.xlabel('n_components')

    plt.savefig('./bic-aic.png')


if __name__ == '__main__':

    '''np.random.seed(42)
    random.seed(42)'''

    # Parse all the input argument
    parser = argparse.ArgumentParser(description='PyTorch Training')

    parser.add_argument('--results-path', type=str, default='./results')
    parser.add_argument('--dataset-name', type=str, default='metalwoz')
    parser.add_argument('--store-id', default='0', type=str)
    parser.add_argument('--data-file-name', default="dialog_init_embedding_100.pkl", type=str)

    parser.add_argument('--epoch', default=100, type=int, help='cluster nums')
    parser.add_argument('--n-clusters', default=39, type=int, help='cluster nums')
    parser.add_argument('--n-clusters_utt', default=100, type=int, help='cluster nums')
    parser.add_argument('--method', default="gmm", type=str)
    parser.add_argument('--calculate-centers', default=True, type=bool)

    parser.add_argument('--type', default='dialog', type=str)

    args = parser.parse_args()

    #np.random.seed(42)

    itr = 0
    epoch = args.epoch

    utt = True
    if utt:
        args.type = 'utterance'
        args.method = 'kmeans'
        file_path = os.path.join(config.RESULT_DIR, args.dataset_name, args.store_id, 'embeddings', 'utterance_epoch_{}_embedding.pkl'.format(epoch))
        file_path = os.path.join(config.RESULT_DIR, args.dataset_name, args.store_id, 'embeddings', 'utt_init_embedding_{}.pkl'.format(epoch))
    else:
        args.type = 'dialog'
        args.method = 'gmm'
        file_path = os.path.join(config.RESULT_DIR, args.dataset_name, args.store_id, 'embeddings', 'dialog_epoch_{}_embedding.pkl'.format(epoch))
        
        #file_path = os.path.join(config.RESULT_DIR, args.dataset_name, args.store_id, 'embeddings', 'dialog_init_embedding_{}.pkl'.format(epoch))

    df = pd.read_pickle(file_path)
    X = np.array(df['embedding'].values.tolist())
    ids = df['session_id'].values.tolist()
    dia_lens = df['dia_lens'].values.tolist()

    tmp, utt_labels = [], []
    for k, v in enumerate(X):
        tmp.append(np.array(v[:dia_lens[k]]))
        for i in range(dia_lens[k]):
            utt_labels.append("{}_{}".format(ids[k], i))

    X = np.concatenate(tmp)

    n_clusters=[10, 20, 30, 40, 50, 60, 70, 80, 100, 110, 120, 130, 140, 150, 160]
    select_by_BIC(X, n_clusters=n_clusters)
    




