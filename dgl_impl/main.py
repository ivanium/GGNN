import argparse
import time
import numpy as np
import numpy.random as nprand

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from dgl import DGLGraph
from dgl.data import register_data_args
# from dgl.data import load_data

from util import load_data

from model import GGNN

def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def main(args):
    data = load_data(args.dataset)
    # data = load_data(args)

    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    train_mask = torch.ByteTensor(data.train_mask)
    val_mask = torch.ByteTensor(data.val_mask)
    test_mask = torch.ByteTensor(data.test_mask)

    num_classes = data.num_labels
    num_edges = data.graph.number_of_edges()
    feature_dim = features.shape[1]
    num_edge_type = 2

    edge_types = Variable((torch.rand([num_edges]).squeeze() > 0.5).long()) # random assign edge type for all edges

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        edge_types = edge_types.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()

    g = DGLGraph(data.graph)
    g.edata['e'] = edge_types

    model = GGNN(g, num_classes, num_edge_type, feature_dim, feature_dim)

    if cuda:
        model = model.cuda()

    loss_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()

        logits = model(features)
        loss = loss_criterion(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc = evaluate(model, features, labels, val_mask)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
                                             acc, num_edges / np.mean(dur) / 1000))

    print()
    acc = evaluate(model, features, labels, test_mask)
    print("Test Accuracy {:.4f}".format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GGNN')
    register_data_args(parser)
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=20,
            help="number of training epochs")
    parser.add_argument("--hidden-dim", type=int, default=50,
            help="gru hidden state dimension")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
            help="Weight for L2 loss")
    args = parser.parse_args()
    print(args)

    main(args)