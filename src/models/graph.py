import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#from torch_geometric.nn import GCNConv, global_mean_pool


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.randn(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.randn(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        # input: [batch, nodes_seq, in_dim]
        # adj  : [batch, nodes_seq, nodes_seq]

        # -> h: [batch, nodes_seq, out_dim]
        h = torch.matmul(input, self.W)

        N = h.size(1)
        bts = h.size(0)

        # -> [batch, nodes_seq, for_each_nodes, out_dim]
        a_input = torch.cat([h.repeat(1, 1, N).view(bts, N * N, -1), h.repeat(1, N, 1)], dim=2).view(bts, N, -1, 2 * self.out_features)

        # -> [batch, nodes_seq, nodes_seq]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        zero_vec = -9e15 * torch.ones_like(e)
        # -> [batch, nodes_seq, nodes_seq]
        attention = torch.where(adj > 0, e, zero_vec)

        attention = F.softmax(attention, dim=2)

        attention = F.dropout(attention, self.dropout, training=self.training)

        # -> h_prime: [batch, nodes_seq, out_dim]
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):

    def __init__(self, n_in_feat=256, n_hid=256, n_out_feat=256, dropout=0.3, alpha=0.2, nheads=8):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(n_in_feat, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        self.attentions2 = [GraphAttentionLayer(n_in_feat, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        for i, attention in enumerate(self.attentions2):
            self.add_module('attention_{}_{}'.format(2, i), attention)

        self.out_att = GraphAttentionLayer(n_hid * nheads, n_out_feat, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj).unsqueeze(2) for att in self.attentions], dim=2)
        x = x.mean(dim=2).squeeze(2)

        x = torch.cat([att(x, adj).unsqueeze(2) for att in self.attentions2], dim=2)
        x = x.mean(dim=2).squeeze(2)

        return x


if __name__ == "__main__":

    gat = GAT(10, 10, 10, 0, 0.2, 8)

    input = torch.ones((2, 4, 10))
    input[:, 0, :] = 0
    input[:, -1, :] = 2

    mask = torch.ones(4, 4)
    mask1 = mask.triu(diagonal=2)
    mask2 = mask.tril(diagonal=-2)
    mask = mask1 + mask2

    adj = mask

    print(adj)

    out = gat(input, adj)

    print(out)
    print(out.size())