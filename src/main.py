import os
import sys
import argparse
import numpy as np
from time import time
from data_loader import load_data
from train import train
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import ShuffleSplit

np.random.seed(555)
parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=30, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=64, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=256, help='dimension of user and entity embeddings')
parser.add_argument('--n_hop', type=int, default=1, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--l2_weight', type=float, default=0.0039, help='weight of l2 regularization')
parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
parser.add_argument('--trainv1_testv2', type=bool, default=False, help='train_on_v1_test_on_v2')

param_name = 'final' # set as the parameter name while adjust parameters
args = parser.parse_args(args=[])
data = load_data(args)  # n_nodea, n_nodeb, n_entity, n_relation, adj_entity（4）, adj_relation（5）
kf = ShuffleSplit(n_splits=10,test_size=0.2,random_state=43)

# reading rating file for split dataset
print('reading sl2id file again for spliting dataset...')
sl2id_file = '../data/sl2id'
if os.path.exists(sl2id_file + '.npy'):
    sl2id_np = np.load(sl2id_file + '.npy')
else:
    sl2id_np = np.loadtxt(sl2id_file + '.txt', dtype=np.int64)
    np.save(sl2id_file + '.npy', sl2id_np)
np.random.shuffle(sl2id_np)

# test data from v2
if (args.trainv1_testv2):
    version2_notin_version1_file = '../data/version2_notin_version1'
    if os.path.exists(version2_notin_version1_file + '.npy'):
        version2_notin_version1_np = np.load(version2_notin_version1_file + '.npy')
    else:
        version2_notin_version1_np = np.loadtxt(version2_notin_version1_file + '.txt', dtype=np.int64)
        np.save(version2_notin_version1_file + '.npy', version2_notin_version1_np)
    np.random.shuffle(version2_notin_version1_np)



k = 1
train_auc_kkf_list = []
train_f1_kkf_list = []
train_aupr_kkf_list = []

eval_auc_kkf_list = []
eval_f1_kkf_list = []
eval_aupr_kkf_list = []

# 观察是否过拟合
test_auc_kkf_list = []
test_f1_kkf_list = []
test_aupr_kkf_list = []

loss_kkf_list = []

for train_data, test_data in kf.split(sl2id_np):
    tf.reset_default_graph()

    if (not args.trainv1_testv2):
        train_data = sl2id_np[train_data]
        test_data = sl2id_np[test_data]

    if (args.trainv1_testv2):
        train_data = sl2id_np
        test_data = version2_notin_version1_np

    # data = load_data(args)  # n, n_item, n_entity, n_relation, train_data(4), eval_data(5), test_data(6), adj_entity（7）, adj_relation（8）
    data = list(data)
    data.append(train_data)
    data.append(test_data)
    data = tuple(data) # n_nodea, n_nodeb, n_entity, n_relation, adj_entity(4), adj_relation(5), train_data(6), test_data(7)
    loss_kf_mean, train_auc_kf_mean, train_f1_kf_mean, train_aupr_kf_mean, eval_auc_kf_mean, eval_f1_kf_mean, eval_aupr_kf_mean, test_auc_kf_mean, test_f1_kf_mean, test_aupr_kf_mean = train(args, data, param_name + '_' + str(k))

    train_auc_kkf_list.append(train_auc_kf_mean)
    train_f1_kkf_list.append(train_f1_kf_mean)
    train_aupr_kkf_list.append(train_aupr_kf_mean)
    eval_auc_kkf_list.append(eval_auc_kf_mean)
    eval_f1_kkf_list.append(eval_f1_kf_mean)
    eval_aupr_kkf_list.append(eval_aupr_kf_mean)
    test_auc_kkf_list.append(test_auc_kf_mean)
    test_f1_kkf_list.append(test_f1_kf_mean)
    test_aupr_kkf_list.append(test_aupr_kf_mean)
    loss_kkf_list.append(loss_kf_mean)
    k = k + 1
    break

train_auc_kkf_mean = np.mean(train_auc_kkf_list)
train_f1_kkf_mean = np.mean(train_f1_kkf_list)
train_aupr_kkf_mean = np.mean(train_aupr_kkf_list)
eval_auc_kkf_mean = np.mean(eval_auc_kkf_list)
eval_f1_kkf_mean = np.mean(eval_f1_kkf_list)
eval_aupr_kkf_mean = np.mean(eval_aupr_kkf_list)

test_auc_kkf_mean = np.mean(test_auc_kkf_list)
test_f1_kkf_mean = np.mean(test_f1_kkf_list)
test_aupr_kkf_mean = np.mean(test_aupr_kkf_list)
loss_kkf_mean = np.mean(loss_kkf_list)

print('final results')
print('train auc_mean: %.4f  train f1_mean: %.4f    train_aupr_mean: %.4f    eval auc_mean: %.4f  eval f1_mean: %.4f    eval_aupr_mean: %.4f    test auc_mean: %.4f  test f1_mean: %.4f  test_aupr_mean: %.4f  loss_mean: %.4f'
    % (train_auc_kkf_mean, train_f1_kkf_mean, train_aupr_kkf_mean, eval_auc_kkf_mean, eval_f1_kkf_mean,
       eval_aupr_kkf_mean, test_auc_kkf_mean, test_f1_kkf_mean, test_aupr_kkf_mean, loss_kkf_mean))
