import torch
import jieba, random, os, sys
import numpy as np
from collections import Counter
import copy
import seaborn as sns
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
#from sklearn.utils.linear_assignment_ import linear_assignment
import config


def init_dirs(dataset_name, store_id):

    if not os.path.isdir(os.path.join(config.RESULT_DIR)):
        os.mkdir(os.path.join(config.RESULT_DIR))

    if not os.path.isdir(os.path.join(config.RESULT_DIR, dataset_name)):
        os.mkdir(os.path.join(config.RESULT_DIR, dataset_name))
        
    if not os.path.isdir(os.path.join(config.RESULT_DIR, dataset_name, store_id)):
        os.mkdir(os.path.join(config.RESULT_DIR, dataset_name, store_id))

    # for store trained model
    if not os.path.isdir(os.path.join(config.RESULT_DIR, dataset_name, store_id, 'models')):
        os.mkdir(os.path.join(config.RESULT_DIR, dataset_name, store_id, 'models'))

    # for store trained model
    if not os.path.isdir(os.path.join(config.RESULT_DIR, dataset_name, store_id, 'embeddings')):
        os.mkdir(os.path.join(config.RESULT_DIR, dataset_name, store_id, 'embeddings'))
    
    # for store trained model
    if not os.path.isdir(os.path.join(config.RESULT_DIR, dataset_name, store_id, 'labels')):
        os.mkdir(os.path.join(config.RESULT_DIR, dataset_name, store_id, 'labels'))
    
    # for store trained model
    if not os.path.isdir(os.path.join(config.RESULT_DIR, dataset_name, store_id, 'centers')):
        os.mkdir(os.path.join(config.RESULT_DIR, dataset_name, store_id, 'centers'))
    
    # for store trained model
    if not os.path.isdir(os.path.join(config.RESULT_DIR, dataset_name, store_id, 'clustering_results')):
        os.mkdir(os.path.join(config.RESULT_DIR, dataset_name, store_id, 'clustering_results'))
    
    # for store trained model
    if not os.path.isdir(os.path.join(config.RESULT_DIR, dataset_name, store_id, 'tsne')):
        os.mkdir(os.path.join(config.RESULT_DIR, dataset_name, store_id, 'tsne'))

    # for store trained model
    if not os.path.isdir(os.path.join(config.RESULT_DIR, dataset_name, store_id, 'dists')):
        os.mkdir(os.path.join(config.RESULT_DIR, dataset_name, store_id, 'dists'))


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.enabled = True


def greedy(greedy_output, tgt_out, voc, loss=None):

    id_2_word = {}
    for k, v in voc.items():
        id_2_word[v] = k

    loss = loss.reshape(-1, loss.shape[-1])
    for i in range(len(greedy_output)):
        print('0. CE loss   :', end='')
        for w in loss[i]:
            print("{:10s}".format(str(round(w, 4))), end='')
        print()
        print('1. Prediction:', end='')
        for w in greedy_output[i]:
            print("{:10s}".format(id_2_word[w]), end='')
        print()
        
        print('2. Truth     :', end='')
        for w in tgt_out[i]:
            print("{:10s}".format(id_2_word[w]), end='')
        print('\n')


def calculate_probs_with_T_distribution(Z: torch.Tensor, centers: torch.Tensor, alpha = 1.0):
    # Z: [es_num x dim]
    # centers: [clu_num x dim]
    
    clus_num = centers.size(0)

    # -> [es_num x clus_num x dim]
    z_i = Z.unsqueeze(1).repeat(1, clus_num, 1)
    # -> [1 x clus_num x dim]
    u_j = centers.unsqueeze(0)
    
    # -> [clus_num, es_num]
    tmp1 = torch.pow(1+torch.pow(z_i-u_j, 2).sum(dim=2)/alpha, -(1+alpha)/2)#.transpose(0, 1)
    # -> [es_num, 1]
    tmp2 = torch.pow(1+torch.pow(z_i-u_j, 2).sum(dim=2)/alpha, -(1+alpha)/2).sum(dim=1).unsqueeze(1)

    # -> [clus_num, es_num]
    res = tmp1 / tmp2

    return res


def draw_heatmap(X, save_name=None):
    fig, axes = plt.subplots(2, 1, figsize=(16, 32))
    sns.heatmap(X, ax=axes[0])
    axes[0].set_title('cluster i to j distance')
    axes[0].set_xlabel('cluster j')
    axes[0].set_ylabel('cluster i')

    for i in range(len(X)):
        X[i] = sorted(X[i])

    sns.heatmap(X, ax=axes[1])
    axes[1].set_title('cluster i to nestest j distance')
    axes[1].set_xlabel('cluster j')
    axes[1].set_ylabel('cluster i')

    plt.savefig(save_name)

def make_mask(lens):
    batch_size = lens.size(0)
    max_len = lens.max()
    mask = torch.ones(batch_size, max_len)
    for i in range(batch_size):
        mask[i][lens[i]:] = 0
    
    return mask

def generate_padding0_mask(lens: torch.LongTensor):

    batch_size = lens.size(0)
    max_len = lens.max()
    mask = torch.ones(batch_size, max_len)
    for i in range(batch_size):
        mask[i][lens[i]:] = 0
    
    mask = mask.float().masked_fill(mask == 0, True).masked_fill(mask == 1, False).bool()

    return mask

def generate_padding_mask(lens:torch.LongTensor, max_len=30):
    batch_size = lens.size(0)
    mask = torch.zeros(batch_size, max_len)
    for i in range(batch_size):
        if lens[i] <= 0:
            mask[i, :] = 1
        else:
            mask[i, lens[i]:] = 1
    
    #mask = mask.float().masked_fill(mask == 0, ).masked_fill(mask == 1, 0).bool()

    return mask.bool()

def generate_square_subsequent_mask(sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == False, float('-inf')).masked_fill(mask == True, float(0.0))
    return mask

def generate_rectangle_subsequent_mask(sz0, sz1, window_size=None):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """

    if window_size is None:
        mask = (torch.triu(torch.ones(sz1, sz0)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == False, float('-inf')).masked_fill(mask == True, float(0.0))
    else:
        pass
    return mask


def generate_low_window_mask_matrix(row, col, window_size=2):
    window_size += 1
    mask = torch.ones(row, col)
    mask1 = mask.tril(diagonal=-window_size)
    mask2 = mask.tril(diagonal=0)

    mask = mask2 - mask1

    return mask.bool()


def generate_window_mask_matrix(row, col, window_size=2):
    window_size += 1
    mask = torch.ones(row, col)
    mask1 = mask.triu(diagonal=window_size)
    mask2 = mask.tril(diagonal=-window_size)
    mask = mask1 + mask2
    return mask.bool()


def label_to_id(labels):
    unique_labels = sorted(list(set(labels)))
    ids = []
    for l in labels:
        ids.append(unique_labels.index(l))
    return ids


def calculate_discount(K=10, mid=8, r=0.8):
    pre = 0
    for i in range(mid):
        pre = pre * r + 1
    aft = 0
    for i in range(mid, K):
        aft  = aft * r + 1
        pre = pre * r

    print('pre', pre)
    print('aft', aft)


if __name__ == "__main__":

    # 1 1 1
    # 0.64 0.8 1 = 2.44
    # 1
    # 0.8 + 1 = 1.8
    # 1.8 * 0.8 + 1 = 2.44

    calculate_discount(K=100, mid=97, r=0.9)

    m = generate_window_mask_matrix(5, 5, window_size=2)

    print(m)