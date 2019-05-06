from __future__ import absolute_import

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import os
import sys

import torch
from torch.autograd import Variable

_urls = {
    'cora': 'dataset/cora_raw.zip',
    'citeseer': 'dataset/citeseer.zip',
    'pubmed': 'dataset/pubmed.zip',
    'cora_binary': 'dataset/cora_binary.zip',
}


def _pickle_load(pkl_file):
    if sys.version_info > (3, 0):
        return pkl.load(pkl_file, encoding='latin1')
    else:
        return pkl.load(pkl_file)


class GraphDataset(object):
    def __init__(self, name):
        self.name = name
        self.dir = '/home/qiaoyf'
        self._load()

    def _load(self):
        """Loads input data from gcn/data directory

        ind.name.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
        ind.name.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
        ind.name.allx => the feature vectors of both labeled and unlabeled training instances
            (a superset of ind.name.x) as scipy.sparse.csr.csr_matrix object;
        ind.name.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
        ind.name.ty => the one-hot labels of the test instances as numpy.ndarray object;
        ind.name.ally => the labels for instances in ind.name.allx as numpy.ndarray object;
        ind.name.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
            object;
        ind.name.test.index => the indices of test instances in graph, for the inductive setting as list object.

        All objects above must be saved using python pickle module.

        :param name: Dataset name
        :return: All data input files loaded (as well the training/test data).
        """
        root = '{}/{}'.format(self.dir, self.name)
        # objnames = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objnames = ['allx', 'ally', 'graph']
        objects = []
        for i in range(len(objnames)):
            with open("{}/ind.{}.{}".format(root, self.name, objnames[i]), 'rb') as f:
                objects.append(_pickle_load(f))

        allx, ally, graph = tuple(objects)
        train_idx_reorder = _parse_index_file(
            "{}/ind.{}.train.index".format(root, self.name))
        test_idx_reorder = _parse_index_file(
            "{}/ind.{}.test.index".format(root, self.name))
        val_idx_reorder = _parse_index_file(
            "{}/ind.{}.val.index".format(root, self.name))

        test_idx_range = np.sort(test_idx_reorder)

        features = sp.vstack(allx).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        graph = nx.DiGraph(nx.from_dict_of_lists(graph))

        onehot_labels = np.vstack(ally)
        onehot_labels[test_idx_reorder, :] = onehot_labels[test_idx_range, :]
        labels = np.argmax(onehot_labels, 1)

        idx_test = test_idx_range.tolist()
        idx_train = train_idx_reorder
        idx_val = val_idx_reorder

        train_mask = _sample_mask(idx_train, labels.shape[0])
        val_mask = _sample_mask(idx_val, labels.shape[0])
        test_mask = _sample_mask(idx_test, labels.shape[0])

        self.graph = Graph(graph)
        self.features = _preprocess_features(features)
        self.labels = labels
        self.onehot_labels = onehot_labels
        self.num_labels = onehot_labels.shape[1]
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask

        print('Finished data loading and preprocessing for '+self.name+".")
        print('  NumNodes: {}'.format(self.graph.num_nodes))
        print('  NumEdges: {}'.format(self.graph.num_edges))
        print('  NumFeats: {}'.format(self.features.shape[1]))
        print('  NumClasses: {}'.format(self.num_labels))
        print('  NumTrainingSamples: {}'.format(
            len(np.nonzero(self.train_mask)[0])))
        print('  NumValidationSamples: {}'.format(
            len(np.nonzero(self.val_mask)[0])))
        print('  NumTestSamples: {}'.format(
            len(np.nonzero(self.test_mask)[0])))

    def __getitem__(self, idx):
        return self

    def __len__(self):
        return 1


def _preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    # limit feature dimension to 20
    return np.array(features.todense())[:, :20]


def _parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def _sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return mask


def load_data(dataset):
    data = GraphDataset(dataset)
    return data


class Graph(object):
    def __init__(self, graph):
        self.edge_list = torch.tensor(list(graph.edges), dtype=torch.long)
        self.src_list = self.edge_list.t()[0]
        self.dst_list = self.edge_list.t()[1]
        self.num_nodes = graph.number_of_nodes()
        self.num_edges = self.edge_list.shape[0]
        # random assign edge type for all edges
        self.edge_types = Variable(
            (torch.rand([self.num_edges]).squeeze() > 0.5))
