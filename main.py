import sys
import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from DGCNN_embedding import DGCNN
from mlp_dropout import MLPClassifier
from sklearn import metrics

sys.path.append('%s/s2v_lib' % os.path.dirname(os.path.realpath(__file__)))
from embedding import EmbedMeanField, EmbedLoopyBP

from util import cmd_args, load_data, load_pet_data, load_minist_data, load_balance_data, load_all_minist_data


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        if cmd_args.gm == 'mean_field':
            model = EmbedMeanField
        elif cmd_args.gm == 'loopy_bp':
            model = EmbedLoopyBP
        elif cmd_args.gm == 'DGCNN':
            model = DGCNN
        else:
            print('unknown gm %s' % cmd_args.gm)
            sys.exit()

        if cmd_args.gm == 'DGCNN':
            self.s2v = model(latent_dim=cmd_args.latent_dim,
                             output_dim=cmd_args.out_dim,
                             num_node_feats=cmd_args.feat_dim + cmd_args.attr_dim,
                             num_edge_feats=0,
                             k=cmd_args.sortpooling_k)
        else:
            self.s2v = model(latent_dim=cmd_args.latent_dim,
                             output_dim=cmd_args.out_dim,
                             num_node_feats=cmd_args.feat_dim,
                             num_edge_feats=0,
                             max_lv=cmd_args.max_lv)
        out_dim = cmd_args.out_dim
        if out_dim == 0:
            if cmd_args.gm == 'DGCNN':
                out_dim = self.s2v.dense_dim
            else:
                out_dim = cmd_args.latent_dim
        self.mlp = MLPClassifier(input_size=out_dim, hidden_size=cmd_args.hidden, num_class=cmd_args.num_class,
                                 with_dropout=cmd_args.dropout)

    def PrepareFeatureLabel(self, batch_graph):
        labels = torch.LongTensor(len(batch_graph))
        n_nodes = 0

        if batch_graph[0].node_tags is not None:
            node_tag_flag = True
            concat_tag = []
        else:
            node_tag_flag = False

        if batch_graph[0].node_features is not None:
            node_feat_flag = True
            concat_feat = []
        else:
            node_feat_flag = False

        for i in range(len(batch_graph)):
            labels[i] = batch_graph[i].label
            n_nodes += batch_graph[i].num_nodes
            if node_tag_flag == True:
                concat_tag += batch_graph[i].node_tags
            if node_feat_flag == True:
                tmp = torch.from_numpy(batch_graph[i].node_features).type('torch.FloatTensor')
                concat_feat.append(tmp)

        if node_tag_flag == True:
            concat_tag = torch.LongTensor(concat_tag).view(-1, 1)
            node_tag = torch.zeros(n_nodes, cmd_args.feat_dim)
            # node_tag.scatter_(1, concat_tag, 1)

        if node_feat_flag == True:
            node_feat = torch.cat(concat_feat, 0)

        if node_feat_flag and node_tag_flag:
            # concatenate one-hot embedding of node tags (node labels) with continuous node features
            node_feat = torch.cat([node_tag.type_as(node_feat), node_feat], 1)
        elif node_feat_flag == False and node_tag_flag == True:
            node_feat = node_tag
        elif node_feat_flag == True and node_tag_flag == False:
            pass
        else:
            node_feat = torch.ones(n_nodes, 1)  # use all-one vector as node features

        if cmd_args.mode == 'gpu':
            node_feat = node_feat.cuda()
            labels = labels.cuda()

        return node_feat, labels

    def forward(self, batch_graph):
        node_feat, labels = self.PrepareFeatureLabel(batch_graph)
        embed = self.s2v(batch_graph, node_feat, None)
        out = self.mlp(embed, labels)
        return out

    def output_features(self, batch_graph):
        node_feat, labels = self.PrepareFeatureLabel(batch_graph)
        embed = self.s2v(batch_graph, node_feat, None)
        return embed, labels


def loop_dataset(g_list, classifier, sample_idxes, optimizer=None, bsize=cmd_args.batch_size):
    total_loss = []
    total_iters = (len(sample_idxes) + (bsize - 1) * (optimizer is None)) // bsize
    all_targets = []
    all_scores = []
    all_pred = []
    n_samples = 0
    for pos in range(total_iters):
        selected_idx = sample_idxes[pos * bsize: (pos + 1) * bsize]

        batch_graph = [g_list[idx] for idx in selected_idx]
        targets = [g_list[idx].label for idx in selected_idx]
        all_targets += targets
        logits, loss, acc = classifier(batch_graph)
        if cmd_args.num_class == 2:
            all_scores.append(logits[:, 1].detach())  # for binary classification
        out_pred = logits.cpu().data.max(1, keepdim=True)[1].numpy().ravel().tolist()
        all_pred += np.asarray(out_pred).reshape(-1).tolist()
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.data.cpu().numpy()

        total_loss.append(np.array([loss, acc]) * len(selected_idx))

        n_samples += len(selected_idx)
    if optimizer is None:
        assert n_samples == len(sample_idxes)
    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / n_samples

    all_targets = np.array(all_targets)
    res_targets = np.asarray(all_targets, dtype=int).reshape(-1)
    res_pred = np.asarray(all_pred).ravel()
    print('---Label---')
    print(res_targets)
    print('---Predict---')
    print(res_pred)
    if cmd_args.num_class == 2:
        all_scores = torch.cat(all_scores).cpu().numpy()
        fpr, tpr, _ = metrics.roc_curve(all_targets, all_scores, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i in range(len(res_targets)):
            if res_pred[i] == 1 and res_targets[i] == 1:
                TP += 1
            elif res_pred[i] == 1 and res_pred[i] != res_targets[i]:
                FP += 1
            elif res_pred[i] == 0 and res_targets[i] == 0:
                TN += 1
            elif res_pred[i] == 0 and res_pred[i] != res_targets[i]:
                FN += 1
        sensitivity = None
        specificity = None
        if TP + FN != 0:
            sensitivity = float(TP) / float(TP + FN)
        if TN + FP != 0:
            specificity = float(TN) / float(TN + FP)
        avg_loss = np.concatenate((avg_loss, [auc], [sensitivity], [specificity]))
    else:
        avg_loss = np.concatenate((avg_loss, [0], [0], [0]))
    return avg_loss


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cmd_args.batch_size = 1
    cmd_args.data = 'balance_pet'  # 'balance_pet' 'minist_all' 'minist'
    cmd_args.plabel = 'ajccLabelSim'  # recurrenceLabel ajccLabelSim survivalLabel
    cmd_args.dropout = True
    cmd_args.batch_norm = True
    cmd_args.extract_features = False
    cmd_args.feat_dim = 0
    cmd_args.fold = 1
    cmd_args.gm = 'DGCNN'
    cmd_args.hidden = 256  # 2048 # 1024 # 512 # 256 # 128
    cmd_args.latent_dim = [32, 32, 32, 1]  # [32, 32, 32, 1]
    cmd_args.learning_rate = 1e-04  # rec 0.001 sur ajcc 0.0001
    cmd_args.max_lv = 4
    cmd_args.mode = 'gpu'
    cmd_args.num_class = 0
    cmd_args.num_epochs = 1000
    cmd_args.out_dim = 0
    cmd_args.printAUC = True
    cmd_args.seed = 1
    cmd_args.sortpooling_k = 0.10
    cmd_args.test_number = 0
    cmd_args.train_number = 100
    cmd_args.n_class = 2
    cmd_args.dropout_r = 0.5

    print(cmd_args)
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    if cmd_args.data == 'pet':
        train_graphs, test_graphs = load_pet_data(cmd_args.train_number, cmd_args.n_class)
    elif cmd_args.data == 'minist':
        train_graphs, test_graphs = load_minist_data(cmd_args.train_number)
    elif cmd_args.data == 'minist_all':
        train_graphs, test_graphs = load_all_minist_data()
    elif cmd_args.data == 'balance_pet':
        train_graphs, test_graphs = load_balance_data(cmd_args.train_number, cmd_args.n_class)
    else:
        train_graphs, test_graphs = load_data()

    print('# train: %d, # test: %d' % (len(train_graphs), len(test_graphs)))

    if cmd_args.sortpooling_k <= 1:
        num_nodes_list = sorted([g.num_nodes for g in train_graphs + test_graphs])
        cmd_args.sortpooling_k = num_nodes_list[int(math.ceil(cmd_args.sortpooling_k * len(num_nodes_list))) - 1]
        cmd_args.sortpooling_k = max(10, cmd_args.sortpooling_k)
        print('k used in SortPooling is: ' + str(cmd_args.sortpooling_k))

    classifier = Classifier()
    print(classifier.s2v, classifier.mlp)
    if cmd_args.mode == 'gpu':
        classifier = classifier.cuda()

    optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)
    #     optimizer = optim.adagrad(classifier.parameters(), lr=cmd_args.learning_rate)

    train_idxes = list(range(len(train_graphs)))
    best_loss = None
    print('test length', len(test_graphs))
    print('train length', len(train_graphs))
    for epoch in range(cmd_args.num_epochs):
        random.shuffle(train_idxes)
        classifier.train()
        avg_loss = loop_dataset(train_graphs, classifier, train_idxes, optimizer=optimizer, bsize=cmd_args.batch_size)
        if not cmd_args.printAUC:
            avg_loss[2] = 0.0
        print('Train of epoch %d: loss %.5f acc %.5f auc %.5f sen %.3f spe %.3f' % (
            epoch, avg_loss[0], avg_loss[1], avg_loss[2], avg_loss[3], avg_loss[4]))

        classifier.eval()
        test_loss = loop_dataset(test_graphs, classifier, list(range(len(test_graphs))))
        if not cmd_args.printAUC:
            test_loss[2] = 0.0
        print('Test of epoch %d: loss %.5f acc %.5f auc %.5f sen %.3f spe %.3f' % (
            epoch, test_loss[0], test_loss[1], test_loss[2], test_loss[3], test_loss[4]))

    with open('acc_results.txt', 'a+') as f:
        f.write(str(test_loss[1]) + '\n')

    if cmd_args.printAUC:
        with open('auc_results.txt', 'a+') as f:
            f.write(str(test_loss[2]) + '\n')

    if cmd_args.extract_features:
        features, labels = classifier.output_features(train_graphs)
        labels = labels.type('torch.FloatTensor')
        np.savetxt('extracted_features_train.txt',
                   torch.cat([labels.unsqueeze(1), features.cpu()], dim=1).detach().numpy(), '%.4f')
        features, labels = classifier.output_features(test_graphs)
        labels = labels.type('torch.FloatTensor')
        np.savetxt('extracted_features_test.txt',
                   torch.cat([labels.unsqueeze(1), features.cpu()], dim=1).detach().numpy(), '%.4f')
