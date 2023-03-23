
import numpy as np
import pandas as pd
import json, os, sys, time
sys.path.append(os.path.abspath("../"))

import matplotlib as mpl
mpl.use('agg')

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
import matplotlib.pyplot as plt
import seaborn as sns

import config, utils, metrics

def draw_scores(data_dict: dict, run_type='dialog', interval=5):
    x, y, methods = {}, {}, {}
    
    for method, scores in data_dict.items():
        for name, score in scores.items():
            res_x, res_y = [], []
            for n, s in sorted(score.items(), key=lambda x: int(x[0])):
                res_x.append(int(n))
                res_y.append(s)
            
            if name not in res_x:
                x[name] = res_x
                y[name] = [res_y]
                methods[name] = [method]
            else:
                #x[name].append(res_x)
                y[name].append(res_y)
                methods[name] = methods
    
    plt.figure(figsize=(16, 9))
    for i, name in enumerate(x.keys()):
        ax = plt.subplot(len(x.keys())//2+1, 2, i+1)
        for y_ in y[name]:
            plt.plot(x[name], y_)
        
        plt.xticks([int(x[name][k]) for k in range(0, len(x[name]), interval)])
        plt.legend(methods[name])
        
        if i % 2 == 0:
            plt.ylabel('score', size=16)
        if i <= 2:
            ax.set_title(name, size=16)

    plt.savefig('{}/{}_scores.png'.format(config.RESULT_DIR, run_type))
    
    

def draw_loss(file_name):
    loss = []
    ppl = []
    kl, dis = [], []
    flg = False
    with open(file_name, 'r') as fp:
        for line in fp:
            if len(line.strip()) and line.strip()[0] == '#' and line.strip()[1].isnumeric():
                flg = True
                continue
            
            if flg and len(line.strip()) and line.strip()[:3] in ['tra', 'val']:
                loss.append(float(line.strip().split(',')[0].split(':')[-1].strip()))
                v = float(line.strip().split(',')[1].split(':')[-1].strip())
                if v > 100000:
                    continue 
                ppl.append(v)

            if flg and len(line.strip()) and line.strip()[:2] == 'kl':
                v = float(line.strip().split(',')[0].split(':')[-1].strip())
                kl.append(v) 
            
            if flg and len(line.strip()) and line.strip()[:3] == 'dis':
                dis.append(float(line.strip().split(',')[0].split(':')[-1].strip()))


    plt.clf()
    plt.figure(figsize=(16, 20))
    num = 30000
    plt.subplot(2, 2, 1)

    plt.grid(c='g')
    plt.plot([i for i in range(len(loss))][:num], loss[:num])
    plt.ylabel('Cross Entropy loss')
    plt.yticks([i for i in range(0, int(max(loss[:num]))+1)])
    #plt.xlabel('# epoch')

    plt.subplot(2, 2, 2)
    plt.grid(c='g')
    plt.plot([i for i in range(len(ppl))][:num], ppl[:num])
    plt.ylabel('perplexity')
    plt.xlabel('# batch')

    plt.subplot(2, 2, 3)
    plt.grid(c='g')
    plt.plot([i for i in range(len(dis))][:num], dis[:num])
    plt.ylabel('MSE')
    plt.xlabel('# batch')


    plt.subplot(2, 2, 4)
    plt.grid(c='g')
    plt.plot([i for i in range(len(kl))][:num], kl[:num])
    plt.ylabel('KL')
    plt.xlabel('# batch')


    plt.savefig('avg_ppl_and_loss_one_epoch_losses_graph_without_detach.png')


    '''plt.clf()
    if kl != []:
        plt.plot([i for i in range(len(kl))], kl)
        plt.savefig('kl2.png')'''


def draw_heatmap(X, save_name=None):
    X_ = []
    for v in X:
        tmp = []
        for vv in X:
            tmp.append(np.sqrt(np.sum(np.square(v-vv))))
        X_.append(tmp)

    X = np.array(X_)

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


def T_SNE(y_pred, X, n_clusters=14, true_labels=[], save_path='', itr=0, epoch=0):


    true_labels = utils.label_to_id(true_labels)

    start = time.time()

    print(len(set(true_labels)), n_clusters)

    #colors = list(set(seaborn.xkcd_rgb.values()))
    colors = ["#0000FF", "#8A2BE2", "#A52A2A", "#000000", "#5F9EA0",
            "#7FFF00", "#D2691E", "#6495ED", "#00008B", "#008B8B",
            "#B8860B", "#A9A9A9", "#006400", "#BDB76B", "#8B008B",
            "#FF8C00", "#00FFFF", "#E9967A", "#FFD700", "#483D8B",
            "#ADFF2F", "#90EE90", "#20B2AA", "#FFA500", "#FFDAB9",
            "#8B4513", "#A0522D", "#008080", "#FFFF00", "#D2B48C",
            "#9ACD32", "#800080", "#EEE8AA", "#800000", "#DEB887",
            "#FF00FF", "#FF6347", "#B0E0E6", "#F5DEB3", "#FFE4E1",
            "#C0C0C0", "#6A5ACD", "#FDF5E6", "#DA70D6", "#ADD8E6",
            "#48D1CC", "#9370DB", "#00FF00", "#E0FFFF", "#4B0082",
            "#696969", "#F5F5F5", "#F4A460", "#C71585", "#7B68EE",
            "#DDA0DD", "#BA55D3", "#CD5C5C", "#2F4F4F", "#FAEBD7"]

    labels_to_color = {}
    for i in range(n_clusters):
        labels_to_color[i] = colors[i]

    

    tsne = TSNE(n_components=2)

    X_tsne = tsne.fit_transform(X)
    #tsne_df = pd.to_pickle(X_tsne, 'tsne_{}_{}.pkl'.format(itr, epoch))

    plt.figure(figsize=(32, 18))
    ax = plt.subplot(121)
    cs = np.array([labels_to_color[v] for v in true_labels])
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cs, marker='.')

    samples = {}
    for k, v in enumerate(true_labels):
        if v in samples and samples[v][0] != 5:
            samples[v][0] += 1
        elif v not in samples:
            samples[v] = [0, k]
    
    samples_ = []
    label_s = []
    for k, v in samples.items():
        samples_.append(X_tsne[v[1]])
        label_s.append(k)
    samples_ = np.array(samples_)

    for i in range(samples_.shape[0]):
        plt.text(samples_[i, 0], samples_[i, 1], str(label_s[i]), fontsize=16)

    ax.set_title('Ground Truth labels', fontsize=24)
    ax = plt.subplot(122)
    cs = np.array([labels_to_color[v] for v in y_pred])
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cs, marker='.')
    ax.set_title('Prediction labels', fontsize=24)
    #plt.savefig('dialog_points.png')

    plt.savefig(save_path)
    print('#Cost: ', time.time()-start)


def T_SNE_ass(y_pred, y_ass, X, n_clusters=14, true_labels=[], save_path='', itr=0, epoch=0):


    true_labels = utils.label_to_id(true_labels)


    start = time.time()

    #colors = list(set(seaborn.xkcd_rgb.values()))
    colors = ["#0000FF", "#8A2BE2", "#A52A2A", "#000000", "#5F9EA0",
            "#7FFF00", "#D2691E", "#6495ED", "#00008B", "#008B8B",
            "#B8860B", "#A9A9A9", "#006400", "#BDB76B", "#8B008B",
            "#FF8C00", "#00FFFF", "#E9967A", "#FFD700", "#483D8B",
            "#ADFF2F", "#90EE90", "#20B2AA", "#FFA500", "#FFDAB9",
            "#8B4513", "#A0522D", "#008080", "#FFFF00", "#D2B48C",
            "#9ACD32", "#800080", "#EEE8AA", "#800000", "#DEB887",
            "#FF00FF", "#FF6347", "#B0E0E6", "#F5DEB3", "#FFE4E1",
            "#C0C0C0", "#6A5ACD", "#FDF5E6", "#DA70D6", "#ADD8E6",
            "#48D1CC", "#9370DB", "#00FF00", "#E0FFFF", "#4B0082",
            "#696969", "#F5F5F5", "#F4A460", "#C71585", "#7B68EE",
            "#DDA0DD", "#BA55D3", "#CD5C5C", "#2F4F4F", "#FAEBD7"]

    labels_to_color = {}
    for i in range(n_clusters):
        labels_to_color[i] = colors[i]

    tsne = TSNE(n_components=2)

    X_tsne = tsne.fit_transform(X)
    #tsne_df = pd.to_pickle(X_tsne, 'tsne_{}_{}.pkl'.format(itr, epoch))

    plt.figure(figsize=(32, 36))
    ax = plt.subplot(221)
    cs = np.array([labels_to_color[v] for v in true_labels])
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cs, marker='.')

    samples = {}
    for k, v in enumerate(true_labels):
        if v in samples and samples[v][0] != 5:
            samples[v][0] += 1
        elif v not in samples:
            samples[v] = [0, k]
    
    samples_ = []
    label_s = []
    for k, v in samples.items():
        samples_.append(X_tsne[v[1]])
        label_s.append(k)
    samples_ = np.array(samples_)

    for i in range(samples_.shape[0]):
        plt.text(samples_[i, 0], samples_[i, 1], str(label_s[i]), fontsize=16)

    ax.set_title('Ground Truth labels', fontsize=24)
    ax = plt.subplot(222)
    cs = np.array([labels_to_color[v] for v in y_pred])
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cs, marker='.')
    ax.set_title('Prediction labels', fontsize=24)
    #plt.savefig('dialog_points.png')

    ax = plt.subplot(224)
    cs = np.array([labels_to_color[v] for v in y_ass])
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cs, marker='.')
    ax.set_title('Prediction global labels', fontsize=24)

    plt.savefig(save_path)
    print('#Cost: ', time.time()-start)

def get_tsne(dataset_name, store_id, data_file_name, n_clusters, clustering_results_file_name, use_ass=False):

    path = os.path.join(config.RESULT_DIR, dataset_name, store_id, 'embeddings', data_file_name)

    data = pd.read_pickle(path)

    path = os.path.join(config.RESULT_DIR, dataset_name, store_id, 'clustering_results', clustering_results_file_name+'.pkl')
    y_true, y_pred = pd.read_pickle(path)
    if use_ass:
        path = os.path.join(config.RESULT_DIR, dataset_name, store_id, 'clustering_results', clustering_results_file_name+'_ass.pkl')
        _, y_ass = pd.read_pickle(path)

    X = np.array(data['embedding'].values.tolist())

    if not os.path.isdir(os.path.join(config.RESULT_DIR, dataset_name, store_id, 'tsne')):
        os.mkdir(os.path.join(config.RESULT_DIR, dataset_name, store_id, 'tsne'))
    save_path = os.path.join(config.RESULT_DIR, dataset_name, store_id, 'tsne', clustering_results_file_name+'.png')

    '''acc, pre_tre = metrics.calculate_acc(y_true, y_pred)
    pre_tre_map = {}
    for i, j in pre_tre:
        pre_tre_map[i] = j
    for i in range(len(y_pred)):
        y_pred[i] = pre_tre_map[y_pred[i]]'''

    if use_ass:
        acc, pre_tre = metrics.calculate_acc(y_true, y_ass)
        pre_tre_map = {}
        for i, j in pre_tre:
            pre_tre_map[i] = j
        for i in range(len(y_ass)):
            y_ass[i] = pre_tre_map[y_ass[i]]
    

    if not use_ass:
        T_SNE(y_pred, X, n_clusters=n_clusters, true_labels=y_true, save_path=save_path)
    else:
        T_SNE_ass(y_pred, y_ass, X, n_clusters=n_clusters, true_labels=y_true, save_path=save_path)
    

if __name__ == "__main__":


    tsne = True
    epoch = 30

    dataset_name = 'sgd_multi'
    store_id = '2'

    n_clusters =59

    if not tsne:

        
        file_path = os.path.join(config.RESULT_DIR, dataset_name, store_id, 'centers', 'centers_{}_dialog.pkl'.format(epoch))
        save_name = os.path.join(config.RESULT_DIR, dataset_name, store_id, 'centers', 'centers_{}_dialog.png'.format(epoch))
        
        file_path = os.path.join(config.RESULT_DIR, dataset_name, store_id, 'centers', 'centers_{}_utterance.pkl'.format(epoch))
        save_name = os.path.join(config.RESULT_DIR, dataset_name, store_id, 'centers', 'centers_{}_utterance.png'.format(epoch))
        
        #file_path = os.path.join(config.RESULT_DIR, dataset_name, store_id, 'centers', 'centers_init_dialog_{}.pkl'.format(epoch))
        #save_name = os.path.join(config.RESULT_DIR, dataset_name, store_id, 'centers', 'centers_init_dialog_{}.png'.format(epoch))
        

        X = pd.read_pickle(file_path)
        #import torch
        #X = torch.randn(100, 256).numpy()

        draw_heatmap(X, save_name)

        exit()

    else:

        itr = 0

        

        clustering_results_file_name = 'true_to_pred_{}_{}'.format(itr, epoch)
        clustering_results_file_name = 'true_to_pred_{}'.format(epoch)

        data_file_name = 'dialog_init_embedding_100.pkl'
        #data_file_name = 'dialog_itr_{}_epoch_{}_embedding.pkl'.format(itr, epoch)
        data_file_name = 'dialog_epoch_{}_embedding.pkl'.format(epoch)
        use_ass = False

        get_tsne(dataset_name, store_id, data_file_name, n_clusters, clustering_results_file_name, use_ass=use_ass)

        exit()

    np.random.seed(0)
    import random
    random.seed(0)

    colors = list(set(seaborn.xkcd_rgb.values()))
    colors.sort()

    colors = ['#000000', '#550f00', "#AA0990", '#FF0000', '#005500', 
                '#0fAAf0', '#00FF00', '#000055', '#0000AA', '#0000FF', 
                '#555500', '#55AA00', '#5500FF', '#55FF00', '#AA0055', 
                '#AA5500', '#AA55FF', '#5555FF', '#FF55FF', '#AF0f50',
                '#05f5f5', '#5a0aa8', '#5f8f08', '#50fc59', '#a0a0a0']
    '''colors = []
    for i in range(0, int('0xffffff', 16), 200):
        c = '{}'.format(str(hex(i))[2:]).ljust(6, '0')
        c = '#' + c
        colors.append(c)
    '''

    colors = ["#0000FF", "#8A2BE2", "#A52A2A", "#000000", "#5F9EA0",
                "#7FFF00", "#D2691E", "#6495ED", "#00008B", "#008B8B",
                "#B8860B", "#A9A9A9", "#006400", "#BDB76B", "#8B008B",
                "#FF8C00", "#00FFFF", "#E9967A", "#FFD700", "#483D8B",
                "#ADFF2F", "#90EE90", "#20B2AA", "#FFA500", "#FFDAB9",
                "#8B4513", "#A0522D", "#008080", "#FFFF00", "#D2B48C",
                "#9ACD32", "#800080", "#EEE8AA", "#800000", "#00FFFF",
                "#FF00FF", "#FF6347", "#B0E0E6", "#F5DEB3", "#FFE4E1"]
    plt.figure(figsize=(32, 18))

    step = 40
    start = 0
    for i in range(start, step+start):
        x = range(10)
        y = np.ones(10)*10 / step + 10*i/step
        print(x)
        print(y, colors[i])
        plt.plot(x, y, linewidth = '20', linestyle='-', color=colors[i], marker='|')
    
    plt.show()
    plt.savefig('test.png')

    '''
    0, 3, 5, 6, 8
    
    '''
