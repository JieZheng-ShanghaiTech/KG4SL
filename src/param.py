import os
import sys
ROOT_DIR = os.path.abspath('E:\KGNN-LS-KGNN-LS-master-edit-v2\src')
sys.path.append(ROOT_DIR)

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


# movie
parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
parser.add_argument('--n_epochs', type=int, default=30, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=32, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=256, help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=1, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--l2_weight', type=float, default=0.0039, help='weight of l2 regularization')
parser.add_argument('--ls_weight', type=float, default=4.5, help='weight of LS regularization')
parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
parser.add_argument('--trainv1_testv2', type=bool, default=False, help='train_on_v1_test_on_v2')

show_loss = True
show_time = False
show_topk = False

t = time()
param_name = 'ls_weight_final'
args = parser.parse_args(args=[])
data = load_data(args)  # n_user, n_item, n_entity, n_relation, train_data(4), eval_data(5), test_data(6), adj_entity（7）, adj_relation（8）
kf = ShuffleSplit(n_splits=5,test_size=0.2,random_state=43)

# reading rating file
rating_file = '../data/' + args.dataset + '/ratings_final'
if os.path.exists(rating_file + '.npy'):
    rating_np = np.load(rating_file + '.npy')
else:
    rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int64)
    np.save(rating_file + '.npy', rating_np)



# param = np.arange(0.5, 8, 0.5)
param = [1e-4, 1e-3, 1e-2, 1e-1, 1,10,100]
param_record = pd.DataFrame(columns=['param', 'loss_mean', 'train_auc_kkf_mean', 'train_f1_kkf_mean', 'train_aupr_kkf_mean', 'eval_auc_kkf_mean', 'eval_f1_kkf_mean', 'eval_aupr_kkf_mean', 'test_auc_kkf_mean','test_f1_kkf_mean','test_aupr_kkf_mean'])
i=1

for p in param:
    args.ls_weight = p
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

    for train_data, test_data in kf.split(rating_np):
        tf.reset_default_graph()
        train_data = rating_np[train_data]
        test_data = rating_np[test_data]
        data = load_data(args)  # n_user, n_item, n_entity, n_relation, train_data(4), eval_data(5), test_data(6), adj_entity（7）, adj_relation（8）
        data = list(data)
        data[4] = train_data
        data[6] = test_data
        data = tuple(data)
        loss_kf_mean, train_auc_kf_mean, train_f1_kf_mean, train_aupr_kf_mean, eval_auc_kf_mean, eval_f1_kf_mean, eval_aupr_kf_mean, test_auc_kf_mean, test_f1_kf_mean, test_aupr_kf_mean = train(args, data, show_loss, show_topk, param_name +'_'+str(round(p,4))+ '_'+ str(k))

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

    print('final results \n')
    print('param %.4f     train auc: %.4f  train f1: %.4f    train_aupr: %.4f    eval auc: %.4f  eval f1: %.4f    eval_aupr: %.4f    test auc: %.4f  test f1: %.4f  test_aupr: %.4f  loss: %.4f'
        % (round(p,4),train_auc_kkf_mean, train_f1_kkf_mean, train_aupr_kkf_mean, eval_auc_kkf_mean, eval_f1_kkf_mean,
           eval_aupr_kkf_mean, test_auc_kkf_mean, test_f1_kkf_mean, test_aupr_kkf_mean, loss_kkf_mean))

    param_record.loc[i] = [p, loss_kkf_mean, train_auc_kkf_mean, train_f1_kkf_mean, train_aupr_kkf_mean, eval_auc_kkf_mean,
                          eval_f1_kkf_mean, eval_aupr_kkf_mean, test_auc_kkf_mean, test_f1_kkf_mean, test_aupr_kkf_mean]
    i = i + 1

param_record.to_csv('../results/'+param_name+'.csv', index=0)

if show_time:
    print('time used: %d s' % (time() - t))



# book
# parser.add_argument('--dataset', type=str, default='book', help='which dataset to use')
# parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
# parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
# parser.add_argument('--dim', type=int, default=64, help='dimension of user and entity embeddings')
# parser.add_argument('--n_iter', type=int, default=2, help='number of iterations when computing entity representation')
# parser.add_argument('--batch_size', type=int, default=256, help='batch size')
# parser.add_argument('--l2_weight', type=float, default=2e-5, help='weight of l2 regularization')
# parser.add_argument('--ls_weight', type=float, default=0.5, help='weight of LS regularization')
# parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')


'''
# music
parser.add_argument('--dataset', type=str, default='music', help='which dataset to use')
parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=16, help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=1, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
parser.add_argument('--ls_weight', type=float, default=0.1, help='weight of LS regularization')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
'''

'''
# restaurant
parser.add_argument('--dataset', type=str, default='restaurant', help='which dataset to use')
parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=4, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=8, help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=2, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=65536, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of l2 regularization')
parser.add_argument('--ls_weight', type=float, default=0.5, help='weight of LS regularization')
parser.add_argument('--lr', type=float, default=2e-2, help='learning rate')
'''
