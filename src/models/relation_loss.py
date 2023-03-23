import torch
import torch.nn as nn
from torch.autograd.function import Function
from torch.nn import functional as F

import sys, random, math
import numpy as np

sys.path.append("..")

import utils


def dot_attention(q, k, v, att_mask=None):
    scaling = float(q.size(-1))**-0.5
    q = q * scaling

    attn_weights = q.bmm(k.transpose(1, 2))

    attn_weights = attn_weights + att_mask.unsqueeze(0)

    attn_weights = F.softmax(attn_weights, 2)

    return attn_weights.bmm(v)


def dot_attention_w(q, k, v, att_mask=None):

    #q = q.repeat(k.size(0), 1, 1)
    scaling = float(q.size(-1))**-0.5
    q = q * scaling

    attn_weights = q.bmm(k.transpose(1, 2))

    attn_weights = attn_weights + att_mask.unsqueeze(0)

    return attn_weights


def get_closest_dist(point, centroids):
    point = point.view(1, -1).repeat(centroids.size(0), 1)

    dis = torch.sqrt(torch.pow(point - centroids, 2).sum(-1))

    return dis.min().item()


# K-means++ style selecting centers
def centers_select(K=0, N=1000, feat_dim=256):
    X = torch.randn(N, feat_dim)
    cluster_centers = random.choice(X).view(1, -1)
    d = [0 for _ in range(len(X))]
    for _ in range(1, K):
        total = 0.0
        for i, point in enumerate(X):
            d[i] = get_closest_dist(point, cluster_centers)
            total += d[i]
        total *= random.random()
        for i, di in enumerate(d):
            total -= di
            if total > 0:
                continue
            cluster_centers = torch.cat([cluster_centers, X[i].view(1, -1)])
            break
    return cluster_centers


class UCRL(nn.Module):

    def __init__(self, n_classes=0, feat_dim=256, device=torch.device('cpu')):
        super(UCRL, self).__init__()
        self.device = device
        centers = centers_select(K=n_classes, N=1000, feat_dim=feat_dim).to(self.device)
        self.centers = nn.Parameter(centers).to(self.device)
        self.soft_max_layer = nn.Linear(feat_dim, n_classes)
        self.bin_layer = nn.Linear(2 * feat_dim, 2)
        self.cos_loss = nn.CosineEmbeddingLoss(margin=0.1, reduction='none')
        self.norm = nn.LayerNorm(feat_dim)

        self.fc1 = nn.Linear(feat_dim, 1024)
        self.fc2 = nn.Linear(1024, feat_dim)

        self.gate_layer = nn.Linear(2 * feat_dim, feat_dim)
        self.margin = 1.0

    def mse_loss(self, feat, target, mask_padding=None):
        loss = torch.pow(feat - target, 2).sum(-1)
        #loss = torch.clamp(loss-self.margin, min=0)

        loss = loss.masked_fill(mask_padding, 0.0)
        loss = loss.sum() / (mask_padding == False).long().sum()

        return loss

    def soft_max_loss(self, feat, target, mask_padding=None):
        out = self.soft_max_layer(feat)
        loss = F.cross_entropy(out.transpose(1, 2), target, reduction='none')

        loss = loss.masked_fill(mask_padding, 0.0)

        loss = loss.sum() / (mask_padding == False).long().sum()

        return loss

    def soft_max_loss_w(self, feat, target, mask_padding=None):
        # feat bts, seq, dim
        # target

        bts, seq_len, dim = feat.size()

        target = target.view(bts, seq_len)
        centers = self.centers.unsqueeze(0).repeat(bts, 1, 1).transpose(1, 2)
        out = feat.bmm(centers)

        loss = F.cross_entropy(out.transpose(1, 2), target, reduction='none')

        loss = loss.masked_fill(mask_padding, 0.0)

        # loss = loss.sum() / (mask_padding == False).long().sum()
        loss = loss.sum(1) / (mask_padding == False).long().sum(1)
        loss = loss.sum() / bts

        return loss

    def bin_sim_loss(self, feat, target, mask_padding=None):

        bts, seq_lens, dim = feat.size()

        feat = feat.unsqueeze(2).repeat(1, 1, self.centers.size(0), 1)

        tgt = self.centers.unsqueeze(0).unsqueeze(0).repeat(feat.size(0), feat.size(1), 1, 1)

        feat = torch.cat([feat, tgt], 3)

        out = self.bin_layer(feat)

        target = torch.zeros(bts, seq_lens, self.centers.size(0), device=feat.device).scatter(2, target.unsqueeze(2), 1.0)

        loss = F.cross_entropy(out.permute(0, 3, 1, 2), target.long(), reduction='none')

        loss = loss.masked_fill(mask_padding.unsqueeze(2), 0.0)

        loss = loss.sum() / (mask_padding == False).long().sum()

        return loss

    def bin_cos_loss(self, feat, target, mask_padding=None):

        bts, seq_lens, dim = feat.size()

        feat = feat.unsqueeze(2).repeat(1, 1, self.centers.size(0), 1)

        tgt = self.centers.unsqueeze(0).unsqueeze(0).repeat(feat.size(0), feat.size(1), 1, 1)

        target = (-1.0 * torch.ones(bts, seq_lens, self.centers.size(0), device=feat.device)).scatter(2, target.unsqueeze(2), 1.0)

        #print(feat.size(), tgt.size(), target.size())

        loss = self.cos_loss(feat.view(-1, feat.size(-1)), tgt.view(-1, tgt.size(-1)), target.view(-1)).view(bts, seq_lens, self.centers.size(0))

        loss = loss.masked_fill(mask_padding.unsqueeze(2), 0.0)

        loss = loss.sum() / (mask_padding == False).long().sum()

        return loss

    def forward(self, feat, labels, mask_padding=None):

        bts, seq_lens, dim = feat.size()

        # Q
        q = feat[:, 1:, :]
        #q = self.fc1(q)
        #q = self.fc2(q)

        centers_batch = self.centers.index_select(0, labels.view(-1).long()).view(bts, seq_lens, dim)

        # K
        k = centers_batch[:, :-1, :]

        att_mask = utils.generate_square_subsequent_mask(k.size(1))  #.to(feat.device)
        att_mask = att_mask.to(self.device)

        V = dot_attention(q, k, k, att_mask=att_mask)

        # add:wq
        #V = V + q * F.sigmoid(self.gate_layer(torch.cat([q, V], dim=-1)))
        # norm
        #V = self.norm(V)
        target = centers_batch[:, 1:, :]
        #loss = self.mse_loss(V, target, mask_padding=mask_padding)
        loss = self.soft_max_loss_w(V, labels.view(bts, seq_lens)[:, 1:], mask_padding=mask_padding)

        #loss = self.bin_sim_loss(V, labels.view(bts, seq_lens)[:, 1:], mask_padding=mask_padding)

        #loss = self.bin_cos_loss(V, labels.view(bts, seq_lens)[:, 1:], mask_padding=mask_padding)
        return loss

    def get_centers(self, labels):
        return self.centers.index_select(0, labels.view(-1).long())


if __name__ == "__main__":

    inputs = torch.rand(2, 3, 10)

    model = UCRL(10, 5, bidirectional=False)
    utt_lens = torch.LongTensor([2, 3])
    labels = torch.LongTensor([[1, 2, 3], [1, 2, 3]])

    out = model(inputs, utt_lens, labels)
