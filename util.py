from __future__ import print_function
import numpy as np
import random
import h5py
import os
import pandas as pd
import scipy.io as si
import cPickle as cp
# import pickle as cp  # python3 compatability
import networkx as nx
import tensorflow as tf
import argparse

cmd_opt = argparse.ArgumentParser(description='Argparser for graph_classification')
cmd_opt.add_argument('-mode', default='cpu', help='cpu/gpu')
cmd_opt.add_argument('-gm', default='mean_field', help='mean_field/loopy_bp')
cmd_opt.add_argument('-data', default=None, help='data folder name')
cmd_opt.add_argument('-batch_size', type=int, default=50, help='minibatch size')
cmd_opt.add_argument('-seed', type=int, default=1, help='seed')
cmd_opt.add_argument('-feat_dim', type=int, default=0, help='dimension of discrete node feature (maximum node tag)')
cmd_opt.add_argument('-num_class', type=int, default=0, help='#classes')
cmd_opt.add_argument('-fold', type=int, default=1, help='fold (1..10)')
cmd_opt.add_argument('-test_number', type=int, default=0,
                     help='if specified, will overwrite -fold and use the last -test_number graphs as testing data')
cmd_opt.add_argument('-num_epochs', type=int, default=1000, help='number of epochs')
cmd_opt.add_argument('-latent_dim', type=str, default='64', help='dimension(s) of latent layers')
cmd_opt.add_argument('-sortpooling_k', type=float, default=30, help='number of nodes kept after SortPooling')
cmd_opt.add_argument('-out_dim', type=int, default=1024, help='s2v output size')
cmd_opt.add_argument('-hidden', type=int, default=100, help='dimension of regression')
cmd_opt.add_argument('-max_lv', type=int, default=4, help='max rounds of message passing')
cmd_opt.add_argument('-learning_rate', type=float, default=0.0001, help='init learning_rate')
cmd_opt.add_argument('-dropout', type=bool, default=False, help='whether add dropout after dense layer')
cmd_opt.add_argument('-printAUC', type=bool, default=False,
                     help='whether to print AUC (for binary classification only)')
cmd_opt.add_argument('-extract_features', type=bool, default=False, help='whether to extract final graph features')

cmd_args, _ = cmd_opt.parse_known_args()

cmd_args.latent_dim = [int(x) for x in cmd_args.latent_dim.split('-')]
if len(cmd_args.latent_dim) == 1:
    cmd_args.latent_dim = cmd_args.latent_dim[0]


class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a numpy array of continuous node features
        '''
        self.num_nodes = len(node_tags)
        self.node_tags = node_tags
        self.label = label
        self.node_features = node_features  # numpy array (node_num * feature_dim)
        self.degs = dict(g.degree).values()

        if len(g.edges()) != 0:
            x, y = zip(*g.edges())
            self.num_edges = len(x)
            self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
            self.edge_pairs[:, 0] = x
            self.edge_pairs[:, 1] = y
            self.edge_pairs = self.edge_pairs.flatten()
        else:
            self.num_edges = 0
            self.edge_pairs = np.array([])


def CalGraph(matlist, filepath, label):
    g_list = []
    for mt in range(len(matlist)):
        matfile = filepath + 'MI_ResOC_suj' + str(matlist[mt]) + '.mat'
        mat = si.loadmat(matfile)
        graph = np.asarray(mat['mi'])[:, :, 5, :]
        graph = (graph > 0.1)

        for i in range(graph.shape[2]):
            g = nx.Graph()
            data = graph[:, :, i]
            for j in range(data.shape[0]):
                g.add_node(j)
                for k in range(data.shape[1]):
                    if data[j][k] != 0:
                        g.add_edge(j, k)
            adj = (data > 0) + 0
            feas1 = np.mean(data, axis=1).reshape(-1, 1)
            feas2 = np.sum(adj, axis=1).reshape(-1, 1)
            node_features = np.hstack((feas1, feas2))
            node_tags = np.ones(data.shape[0]).tolist()
            l = label
            assert len(g) == data.shape[0]
            g_list.append(S2VGraph(g, l, node_tags, node_features))
    return g_list


def load_data():
    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('data/%s/%s.txt' % (cmd_args.data, cmd_args.data), 'r') as f:
        # with open('data/%s/%s.txt' % ('DD', 'DD'), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            # assert len(g.edges()) * 2 == n_edges  (some graphs in COLLAB have self-loops, ignored here)
            assert len(g) == n
            g_list.append(S2VGraph(g, l, node_tags, node_features))
    for g in g_list:
        g.label = label_dict[g.label]
    cmd_args.num_class = len(label_dict)
    cmd_args.feat_dim = len(feat_dict)  # maximum node label (tag)
    if node_feature_flag == True:
        cmd_args.attr_dim = node_features.shape[1]  # dim of node features (attributes)
    else:
        cmd_args.attr_dim = 0

    print('# classes: %d' % cmd_args.num_class)
    print('# maximum node tag: %d' % cmd_args.feat_dim)

    if cmd_args.test_number == 0:
        train_idxes = np.loadtxt('data/%s/10fold_idx/train_idx-%d.txt' % (cmd_args.data, cmd_args.fold),
                                 dtype=np.int32).tolist()
        test_idxes = np.loadtxt('data/%s/10fold_idx/test_idx-%d.txt' % (cmd_args.data, cmd_args.fold),
                                dtype=np.int32).tolist()
        return [g_list[i] for i in train_idxes], [g_list[i] for i in test_idxes]
    else:
        return g_list[: n_g - cmd_args.test_number], g_list[n_g - cmd_args.test_number:]


def from_mat():
    filepath = '../lung_data.mat'
    f = h5py.File(filepath)

    graphdata = f['lung_data']['graph'][0]  # affinityGraph afGraph02 afGraph08 nGraph
    graphs = []
    for g in graphdata:
        graphs.append(np.array(f[g]))
    graphs = np.asarray(graphs)
    gdata = f['lung_data']['graphLabel'][0]
    glabels = []
    for g in gdata:
        glabels.append(np.array(f[g]))
    # len(glabels) = 135
    hdata = f['lung_data']['heterogeneityLabel'][0]
    hlabels = []
    for h in hdata:
        hlabels.append(np.array(f[h]))
    # len(hlabels) = 135
    labeldata = f['lung_data'][cmd_args.plabel][0]
    labels = []
    for l in labeldata:
        labels.append(np.array(f[l]))
    labels = np.asarray(labels).reshape(-1)
    return graphs, labels, hlabels, glabels


def load_pet_data(train_num, class_num=2):
    print('loading pet data...')
    g_list = []
    data, label, hlabel, glabel = from_mat()
    node_features = []
    for i in range(data.shape[0]):
        g = nx.Graph()
        for j in range(data[i].shape[0]):
            g.add_node(j)
            for k in range(data[i].shape[1]):
                if data[i][j][k] != 0:
                    g.add_edge(j, k)
        adj = (data[i] > 0) + 0
        feas1 = glabel[i].reshape(-1, 1)
        feas2 = hlabel[i].reshape(-1, 1)
        feas3 = np.mean(data[i], axis=1).reshape(-1, 1)
        feas4 = np.sum(adj, axis=1).reshape(-1, 1)
        node_features = np.hstack((feas1, feas2, feas3, feas4))
        node_tags = np.ones(data[i].shape[0]).tolist()
        l = label[i]
        if l == 2:
            continue
        assert len(g) == data[i].shape[0]
        g_list.append(S2VGraph(g, l, node_tags, node_features))
    cmd_args.num_class = class_num
    cmd_args.feat_dim = 1
    cmd_args.attr_dim = node_features.shape[1]
    print('# classes: %d' % cmd_args.num_class)
    print('# maximum node tag: %d' % cmd_args.feat_dim)
    idx = np.random.permutation(len(g_list))
    train_idx = idx[0:train_num]
    test_idx = idx[train_num:]
    return [g_list[i] for i in train_idx], [g_list[i] for i in test_idx]


def load_balance_data(train_num, class_num=2):
    train_datas, test_datas = load_pet_data(train_num, class_num)

    train_data = [[] for i in range(class_num)]
    for i in range(len(train_datas)):
        train_data[int(train_datas[i].label)].append(train_datas[i])
    maxtrainlen = 0
    maxtrainidx = 0
    for i in range(len(train_data)):
        if len(train_data[i]) > maxtrainlen:
            maxtrainlen = len(train_data[i])
            maxtrainidx = i
    all_train_data = []
    for i in range(maxtrainlen):
        for j in range(len(train_data)):
            if j == maxtrainidx:
                all_train_data.append(train_data[j][i])
            else:
                tempidx = random.randint(0, len(train_data[j]) - 1)
                all_train_data.append(train_data[j][tempidx])
    print('all train data %d' % len(all_train_data))

    test_data = [[] for i in range(class_num)]
    for i in range(len(test_datas)):
        test_data[int(test_datas[i].label)].append(test_datas[i])
    maxtestlen = 0
    maxtestidx = 0
    for i in range(len(test_data)):
        if len(test_data[i]) > maxtestlen:
            maxtestlen = len(test_data[i])
            maxtestidx = i
    all_test_data = []
    for i in range(maxtestlen):
        for j in range(len(test_data)):
            if j == maxtestidx:
                all_test_data.append(test_data[j][i])
            else:
                tempidx = random.randint(0, len(test_data[j]) - 1)
                all_test_data.append(test_data[j][tempidx])
    print('all test data %d' % len(all_test_data))

    return all_train_data, all_test_data


def construct_graph(x):
    mask = [[-2, -2], [-2, -1], [-2, 0], [-2, 1], [-2, 2],
            [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
            [0, -2], [0, -1], [0, 1], [0, 2],
            [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
            [2, -2], [2, -1], [2, 0], [2, 1], [2, 2]]
    map = np.zeros((28, 28))
    nNodes = 0
    for i in range(28):
        for j in range(28):
            if x[i][j] != 0:
                nNodes = nNodes + 1
                map[i][j] = nNodes

    graph = np.zeros((nNodes, nNodes))
    graph_label = np.zeros((nNodes, 1))
    for i in range(28):
        for j in range(28):
            if map[i][j] != 0:
                node0 = map[i][j]
                pixel0 = x[i][j]
                for t in range(24):
                    ii = i + mask[t][0]
                    jj = j + mask[t][1]
                    if (ii >= 0 and ii <= 27 and jj >= 0 and jj <= 27):
                        if map[ii][jj] != 0:
                            node1 = map[ii][jj]
                            pixel1 = x[ii][jj]
                            if (abs(mask[t][0]) == 2 or abs(mask[t][1]) == 2):
                                coeff = 2
                            else:
                                coeff = 1
                            graph[int(node0 - 1)][int(node1 - 1)] = coeff * abs(int(pixel0) - int(pixel1))
                            graph_label[int(node0 - 1)][0] = pixel0
    return graph, graph_label


def load_all_minist_data():
    print('loading all minist data...')
    g_list_train = []
    g_list_test = []
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    for i in range(x_train.shape[0]):
        g = nx.Graph()
        matrix, feas1 = construct_graph(x_train[i])
        for j in range(matrix.shape[0]):
            g.add_node(j)
            for k in range(matrix.shape[1]):
                if matrix[j][k] != 0:
                    g.add_edge(j, k)
        adj = (matrix > 0) + 0
        feas2 = np.mean(matrix, axis=1).reshape(-1, 1)
        feas3 = np.sum(adj, axis=1).reshape(-1, 1)
        node_features = np.hstack((feas1, feas2, feas3))
        node_tags = np.ones(matrix.shape[0]).tolist()
        l = y_train[i]
        assert len(g) == matrix.shape[0]
        g_list_train.append(S2VGraph(g, l, node_tags, node_features))

    for i in range(x_test.shape[0]):
        g = nx.Graph()
        matrix, feas1 = construct_graph(x_test[i])
        for j in range(matrix.shape[0]):
            g.add_node(j)
            for k in range(matrix.shape[1]):
                if matrix[j][k] != 0:
                    g.add_edge(j, k)
        adj = (matrix > 0) + 0
        feas2 = np.mean(matrix, axis=1).reshape(-1, 1)
        feas3 = np.sum(adj, axis=1).reshape(-1, 1)
        node_features = np.hstack((feas1, feas2, feas3))
        node_tags = np.ones(matrix.shape[0]).tolist()
        l = y_test[i]
        assert len(g) == matrix.shape[0]
        g_list_test.append(S2VGraph(g, l, node_tags, node_features))
    cmd_args.num_class = 10
    cmd_args.feat_dim = 1
    cmd_args.attr_dim = node_features.shape[1]
    print('# classes: %d' % cmd_args.num_class)
    print('# maximum node tag: %d' % cmd_args.feat_dim)
    return g_list_train, g_list_test


def load_minist_data(train_num):
    print('loading minist data...')
    g_list = []
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x1_idx = (y_train == 1)
    x0_idx = (y_train == 0)
    x1 = x_train[x1_idx]
    x0 = x_train[x0_idx]
    x = []
    y = []
    for i in range(train_num):
        x.append(x1[i])
        y.append(1)
        x.append(x0[i])
        y.append(0)
    for i in range(len(x)):
        g = nx.Graph()
        matrix, feas1 = construct_graph(x[i])
        for j in range(matrix.shape[0]):
            g.add_node(j)
            for k in range(matrix.shape[1]):
                if matrix[j][k] != 0:
                    g.add_edge(j, k)
        adj = (matrix > 0) + 0
        feas2 = np.mean(matrix, axis=1).reshape(-1, 1)
        feas3 = np.sum(adj, axis=1).reshape(-1, 1)
        node_features = np.hstack((feas1, feas2, feas3))
        node_tags = np.ones(matrix.shape[0]).tolist()
        l = y[i]
        assert len(g) == matrix.shape[0]
        g_list.append(S2VGraph(g, l, node_tags, node_features))
    cmd_args.num_class = 2
    cmd_args.feat_dim = 1
    cmd_args.attr_dim = node_features.shape[1]
    print('# classes: %d' % cmd_args.num_class)
    print('# maximum node tag: %d' % cmd_args.feat_dim)
    return g_list[0: train_num], g_list[train_num:]


def main():
    load_minist_data(100)


if __name__ == '__main__':
    main()
