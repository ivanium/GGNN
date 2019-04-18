#!/usr/bin/env python
# encoding: utf-8
# File Name: layer.py
# Author: Jiezhong Qiu
# Create Time: 2019/01/09 10:58
# TODO:

import torch
import torch.nn as nn
import dgl
import dgl.function as fn


class MPNNLayer(nn.Module):
    def __init__(self, edge_net_units=[10, 10], msg_dim=10, hidden_dim=10):
        self.msg_dim = msg_dim
        self.hidden_dim = hidden_dim
        edge_net_units = edge_net_units + [msg_dim * hidden_dim, ]
        edge_net = []
        for i in range(1, len(edge_net_units)):
            edge_net.append(
                nn.Linear(edge_net_units[i-1], edge_net_units[i]))
            edge_net.append(nn.ReLU())
        self.edge_net = nn.Sequential(edge_net)

        self.gru = nn.GRU(input_size=msg_dim,
                          hidden_size=hidden_dim)

    def edge_net_msg(self, edge):
        A = self.edge_net(edge.data['e']).view(-1,
                                               self.msg_dim, self.hidden_dim)
        msg = torch.bmm(A, edge.src['h'])
        return {'msg': msg}

    def apply_func(self, nodes):
        return {'h': self.gru(
            nodes.data['m'].unsqueeze(0),
            nodes.data['h'].unsqueeze(0))}

    def forward(self, g):
        g.update_all(message_func=self.edge_net_msg,
                     reduce_func=fn.sum(msg='msg', out='m'),
                     apply_node_func=self.apply_func)


class GGNNLayer(nn.Module):
    def __init__(self, edge_net_units=[10, 10], msg_dim=10, hidden_dim=10,
                 num_edge_type=10):
        self.msg_dim = msg_dim
        self.hidden_dim = hidden_dim
        self.edge_matrix = nn.Embedding(num_embeddings=num_edge_type,
                                        embedding_dim=msg_dim * hidden_dim)
        self.gru = nn.GRU(input_size=msg_dim,
                          hidden_size=hidden_dim)

    def ggnn_msg(self, edge):
        A = self.edge_matrix(
            edge.data['e']).view(-1, self.msg_dim, self.hidden_dim)
        msg = torch.bmm(A, edge.src['h'])
        return {'msg': msg}

    def apply_func(self, nodes):
        return {'h': self.gru(
            nodes.data['msg'].unsqueeze(0),
            nodes.data['h'].unsqueeze(0))}

    def forward(self, g):
        g.update_all(message_func=self.ggnn_msg,
                     reduce_func=fn.sum(msg='msg', out='m'),
                     apply_node_func=self.apply_func)


class GGNNReadout(nn.Module):
    def __init__(self, i_net_units=[10, 10], j_net_units=[10, 10]):
        i_net = []
        for i in range(1, len(i_net_units)):
            i_net.append(nn.Linear(i_net_units[i-1], i_net_units[i]))
            if i+1 < len(i_net_units):
                i_net.append(nn.ReLu())
        self.i_net.append(nn.Sigmoid())
        self.i_net = nn.Sequential(i_net)

        j_net = []
        for i in range(1, len(j_net_units)):
            j_net.append(nn.Linear(j_net_units[i-1], j_net_units[i]))
            if i+1 < len(j_net_units):
                j_net.append(nn.ReLU())
        self.j_net = nn.Sequential(j_net)

    def forward(self, g):
        h0 = g.ndata['h0']
        hT = g.ndata['h']

        i_out = self.i_net(torch.cat((h0, hT), dim=-1))
        j_out = self.j_net(hT)
        g.ndata['r'] = i_out * j_out
        return dgl.sum_nodes(g, feat='r')


class Set2SetReadout(nn.Module):
    def __init__(self):
        raise NotImplementedError
