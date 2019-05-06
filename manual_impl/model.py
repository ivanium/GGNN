import math
import torch
import torch.nn as nn
from torch_scatter import scatter_add
from torch.autograd import Variable
import time


class GGNN(nn.Module):
    def __init__(self,
                 g,
                 num_classes,
                 num_edge_type,
                 msg_dim=10,
                 hidden_dim=10):
        super(GGNN, self).__init__()
        self.g = g
        self.msg_dim = msg_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.edge_matrix = []
        for i in range(num_edge_type):
            self.edge_matrix.append(nn.Linear(hidden_dim, msg_dim))
        self.gru = nn.GRU(input_size=msg_dim, hidden_size=hidden_dim)
        # Output Model
        self.out = nn.Linear(hidden_dim, num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, features):
        typed_srcs, typed_dsts = [], []
        typed_srcs.append(self.g.src_list[self.g.edge_types])
        typed_srcs.append(self.g.src_list[self.g.edge_types == 0])
        typed_dsts.append(self.g.dst_list[self.g.edge_types])
        typed_dsts.append(self.g.dst_list[self.g.edge_types == 0])
        # filter features according to edge type
        typed_features = []
        for i in range(2):
            typed_features.append(features[typed_srcs[i]])

        # num(edge type) times dense matrix multiply
        typed_msgs = []
        for i in range(2):
            typed_msgs.append(self.edge_matrix[i](typed_features[i]))

        # message passing && reduce message
        reduced_msgs = features.new_empty(size=features.shape).t()
        scatter_add(typed_msgs[0].t(), typed_dsts[0].unsqueeze(
            1).expand(typed_msgs[0].shape).t(), out=reduced_msgs)
        reduced_msgs = reduced_msgs + \
            scatter_add(typed_msgs[1].t(), typed_dsts[1].unsqueeze(
                1).expand(typed_msgs[1].shape).t(), dim_size=self.g.num_nodes)

        # batched messages and features feed into gru unit
        _, h = self.gru(
            reduced_msgs.t().unsqueeze(0),
            features.unsqueeze(0)
        )

        # output layer
        h = self.out(h[0])
        return h
