import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score
from collections import defaultdict, Counter
#from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment

calculate_nmi = normalized_mutual_info_score
calculate_ari = adjusted_rand_score


def label_to_id(labels):
    unique_labels = sorted(list(set(labels)))
    ids = []
    for l in labels:
        ids.append(unique_labels.index(l))
    return ids


def calculate_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """

    y_true = label_to_id(y_true)
    y_pred = label_to_id(y_pred)

    pairs = sorted(list(zip(y_true, y_pred)), key=lambda x: x[1])
    y_true, y_pred = np.array([v[0] for v in pairs]), np.array([v[1] for v in pairs])

    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.float64)

    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    #np.random.seed(42)
    # w = w + np.random.randn(w.shape[0], w.shape[1])*1e-4
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    '''print((w.max()-w))
    print((w.max()-w).shape)'''

    ind = np.concatenate([row_ind.reshape(-1, 1), col_ind.reshape(-1, 1)], axis=1)

    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def calculate_purity(y_true, y_pred):
    Cs = {}
    for i in range(len(y_pred)):
        if y_pred[i] not in Cs:
            Cs[y_pred[i]] = [y_true[i]]
        else:
            Cs[y_pred[i]].append(y_true[i])

    res = {}
    correct_nums = 0
    for k, v in Cs.items():
        if v == []:
            res[k] = (str(k), 1)
            continue
        c = Counter(v)
        probs = {}
        for kk, vv in c.items():
            probs[kk] = vv
        res[k] = max(probs.items(), key=lambda x: x[1])
        correct_nums += res[k][1]

    return correct_nums / len(y_true)


def metric(y_true, y_pred):
    acc = calculate_acc(y_true, y_pred)
    purity = calculate_purity(y_true, y_pred)
    nmi = calculate_nmi(y_true, y_pred)
    ari = calculate_ari(y_true, y_pred)

    print("==>     acc      purity      nmi       ari")
    print("==>###  {:.5f}   {:.5f}      {:.5f}    {:.5f}".format(acc, purity, nmi, ari))


def metric_from_ids(id_to_true, id_to_pred):
    y_true, y_pred = [], []
    for k, v in id_to_pred.items():
        y_true.append(id_to_true[str(k)])
        y_pred.append(v)

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    acc = calculate_acc(y_true, y_pred)

    purity = calculate_purity(y_true, y_pred)

    nmi = calculate_nmi(y_true, y_pred)
    ari = calculate_ari(y_true, y_pred)

    print("==> acc: {:.5f},  purity: {:.5f},  nmi: {:.5f}, ari: {:.5f} <==".format(acc, purity, nmi, ari))


def metric_(data_file):

    data = pd.read_pickle(data_file)

    y_true, y_pred = data

    acc = calculate_acc(y_true, y_pred)
    purity = calculate_purity(y_true, y_pred)

    nmi = calculate_nmi(y_true, y_pred)
    ari = calculate_ari(y_true, y_pred)

    print("==> acc: {:.5f},  purity: {:.5f},  nmi: {:.5f}, ari: {:.5f} <==".format(acc, purity, nmi, ari))
