import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class NodeApplyModule(nn.Module):
    def __init__(self, out_feats, activation=None, bias=True):
        super(NodeApplyModule, self).__init__()
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, nodes):
        h = nodes.data['h']
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        return {'h': h}


class GGNN(nn.Module):
    def __init__(self,
                 g,
                 n_classes,
                 msg_dim=10,
                 hidden_dim=10):
        super(GGNN, self).__init__()
        self.g = g
        self.msg_dim = msg_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.edge_matrix = nn.Embedding(
            num_embeddings=num_edge_type, embedding_dim=msg_dim * hidden_dim)
        self.gru = nn.GRU(input_size=msg_dim, hidden_size=hidden_dim)
        # Output Model
        self.out = nn.Linear(hidden_dim, n_classes)

    def ggnn_msg(self, edge):
        A = self.edge_matrix(
            edge.data['e']).view(-1, self.msg_dim, self.hidden_dim)
        msg = torch.dmm(A, edge.src['h'])
        return {'msg': msg}

    def apply_func(self, nodes):
        return {'h': self.gru(
            node.data['msg'].unsqueeze(0),
            node.data['h'].unsqueeze(0)
        )}

    def reset_parameters(self):
        pass

    def forward(self, features):
        self.g.ndata['h'] = features
        self.g.update_all(message_func=self.ggnn_msg,
                          reduce_func=fn.sum(msg='msg', out='m'),
                          apply_node_func=self.apply_func)
        h = self.g.ndata.pop('h')
        return h


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        lables = lables[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)
