import torch, json, sys

sys.path.append("..")
import numpy as np
from scipy.optimize import linear_sum_assignment

import utils


class Statistic(object):
    def __init__(self, n_samples, n_clusters, discount=1e-5, device=None):

        self.device = device

        self.itr = 0
        self.discount = discount

        self.n_samples = n_samples
        self.n_clusters = n_clusters

        self.matrix = np.zeros((self.n_samples, self.n_clusters))
        self.assignment = np.zeros((self.n_samples, ))
        # self.current_labels = {}

    def get_assignment(self):
        return self.assignment

    def best_map(self, y_true, y_pred):

        pairs = sorted(list(zip(y_true, y_pred)), key=lambda x: x[1])
        y_true, y_pred = np.array([v[0] for v in pairs]), np.array([v[1] for v in pairs])

        y_true = y_true.astype(np.int64)
        assert y_pred.size == y_true.size

        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1

        row_ind, col_ind = linear_sum_assignment(w.max() - w)
        ind = np.concatenate([row_ind.reshape(-1, 1), col_ind.reshape(-1, 1)], axis=1)

        acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
        # [pred, true]
        return ind, acc

    def update_labels(self, new_labels):
        self.matrix *= self.discount

        if self.itr == 0:
            self.id_to_index = {}
            for i, k in enumerate(sorted(list(set(new_labels)))):
                self.id_to_index[str(k)] = i

            # init record metrix
            for k, v in new_labels.items():
                self.matrix[self.id_to_index[str(k)]][int(v)] += 1

            _, acc = self.best_map(self.assignment, np.argmax(self.matrix+np.random.randn(self.matrix.shape[0], self.matrix.shape[1])*1e-4, axis=1))

        else:
            # Align current clustering assignment with current recorded assignment
            new_assignment = np.zeros_like(self.assignment)
            for k, v in new_labels.items():
                new_assignment[self.id_to_index[str(k)]] = v

            pre_to_true_list, acc = self.best_map(self.assignment, new_assignment)
            pre_to_true_map = {}
            for pre, tre in pre_to_true_list:
                pre_to_true_map[pre] = tre

            # pred labels map to true labels for each sample
            for i in range(len(new_labels)):
                v = new_assignment[i]
                self.matrix[i, pre_to_true_map[v]] += 1

        # update assgignment
        self.assignment = np.argmax(self.matrix+np.random.randn(self.matrix.shape[0], self.matrix.shape[1])*1e-4, axis=1)
        self.itr += 1

        current_labels = {}
        for k, v in self.id_to_index.items():
            current_labels[str(k)] = int(self.assignment[v])

        return current_labels, acc


if __name__ == "__main__":
    stat = Statistic(5, 3, discount=0.99, id_voc={'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4})

    print(stat.get_assignment())
    print(stat.labels)

    new_labels = {'a': 0, 'b': 1, 'c': 2, 'd': 1, 'e': 0}

    stat.update(new_labels)

    print(stat.get_dist())
    print(stat.statistic)
    print(stat.labels)

    stat.update(news_dist, ids)
    print(stat.get_dist())
    print(stat.statistic)

    print(stat.labels)

    news_dist = np.array([1, 0, 2, 0])
    ids = ['b', 'a', 'c', 'd']

    stat.update(news_dist, ids)
    print(stat.get_dist())
    print(stat.statistic)

    print(stat.labels)

    news_dist = np.array([1, 0, 2, 0])
    ids = ['b', 'a', 'c', 'd']

    stat.update(news_dist, ids)
    print(stat.get_dist())
    print(stat.statistic)

    print(stat.labels)

    exit()
