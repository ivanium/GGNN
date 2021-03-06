import math
import torch
import torch.nn as nn

import dgl
import dgl.function as fn
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
        self.edge_matrix = nn.Embedding(
            num_embeddings=num_edge_type, embedding_dim=msg_dim * hidden_dim)
        self.gru = nn.GRU(input_size=msg_dim, hidden_size=hidden_dim)
        # Output Model
        self.out = nn.Linear(hidden_dim, num_classes)
        self.reset_parameters()

    def ggnn_msg(self, edges):
        A = self.edge_matrix(
            edges.data['e']).view(-1, self.msg_dim, self.hidden_dim)
        msg = torch.bmm(A, edges.src['h'].unsqueeze(2))
        return {'msg': msg}

    def reduce_sum(self, msg, out):
        res = fn.sum(msg, out)
        return res

    def apply_func(self, nodes):
        _, h = self.gru(
            nodes.data['m'].squeeze(2).unsqueeze(0),
            nodes.data['h'].unsqueeze(0)
        )
        return {'h': h[0]}

    def reset_parameters(self):
        pass
        # self.edge_matrix.data.uniform(0.0, 0.02)
        # self.gru.data.uniform(0.0, 0.02)

    def forward(self, features):
        self.g.ndata['h'] = features
        stt = time.time()
        self.g.update_all(message_func=self.ggnn_msg,
                        #   reduce_func=fn.sum(msg='msg', out='m'),
                          reduce_func=self.reduce_sum(msg='msg', out='m'),
                          apply_node_func=self.apply_func)
        end = time.time()
        print("total", end-stt)
        h = self.g.ndata.pop('h')
        h = self.out(h)
        return h
