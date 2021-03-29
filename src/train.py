import tensorflow as tf
import numpy as np
from model import KG4SL
import pandas as pd
from sklearn.model_selection import ShuffleSplit
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def reindexid2geneName(test_data):
    entity2id = pd.read_csv('../data/entity2id.txt', sep='\t')
    entity_dict = {}
    for index, row in entity2id.iterrows():
        entity_dict[row['b']] = row['a']

    test_data.columns = ['gene_a', 'gene_b', 'label']

    db_ida = []
    db_idb = []
    for index, row in test_data.iterrows():
        db_ida.append(entity_dict[row['gene_a']])
        db_idb.append(entity_dict[row['gene_b']])
    test_data['db_ida'] = db_ida
    test_data['db_idb'] = db_idb

    dbid2name = pd.read_csv('../data/dbid2name.csv', sep=',', header=0)
    id2name_dict = {}
    for index, row in dbid2name.iterrows():
        id2name_dict[row['_id']] = row['name']

    name_a = []
    name_b = []
    for index, row in test_data.iterrows():
        name_a.append(id2name_dict[row['db_ida']])
        name_b.append(id2name_dict[row['db_idb']])
    test_data['name_a'] = name_a
    test_data['name_b'] = name_b

    col_order = ['gene_a', 'gene_b', 'db_ida', 'db_idb', 'name_a', 'name_b', 'label']
    test_data = test_data[col_order]

    return test_data

def train(args, data, string):

    # data: # n_nodea(0), n_nodeb(1), n_entity(2), n_relation(3), adj_entity(4), adj_relation(5), train_data(6), test_data(7)
    n_nodea, n_nodeb, n_entity, n_relation = data[0], data[1], data[2], data[3]
    adj_entity, adj_relation = data[4], data[5]

    test_data = data[7]
    
    test_data_mapping = reindexid2geneName(pd.DataFrame(test_data)) # Add gene names
    
    pd.DataFrame(test_data).to_csv('../results/test_data_' + string + '.csv',header=False, index=False)
    test_data_mapping.to_csv('../results/test_data_mapping_' + string + '.csv', header=0, index=False)

    kf = ShuffleSplit(n_splits=9,test_size=0.2,random_state=43)
    cross_validation = 1
    train_auc_kf_list = []
    train_f1_kf_list = []
    train_aupr_kf_list = []

    eval_auc_kf_list = []
    eval_f1_kf_list = []
    eval_aupr_kf_list = []

    test_auc_kf_list = []
    test_f1_kf_list = []
    test_aupr_kf_list = []

    loss_kf_list = []
    loss_curve = pd.DataFrame(columns=['epoch', 'loss', 'train_auc', 'train_f1', 'train_aupr', 'eval_auc', 'eval_f1', 'eval_aupr', 'test_auc','test_f1', 'test_aupr'])
    kk=1

    for train_kf, eval_kf in kf.split(data[6]):
        if (cross_validation == 6):
            print(str(cross_validation-1)+' cross validation stop!')
            break
        else:
            tf.reset_default_graph()

            train_data = data[6][train_kf]
            eval_data = data[6][eval_kf]
            pd.DataFrame(train_data).to_csv('../results/train_data_' + string + '_' + str(kk) + '.csv',header=False, index=False)
            pd.DataFrame(eval_data).to_csv('../results/eval_data_'+ string + '_' + str(kk) + '.csv',header=False, index=False)

            model = KG4SL(args,n_entity, n_relation, adj_entity, adj_relation)

            saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                best_loss_flag = 1000000
                early_stopping_flag = 2
                best_eval_auc_flag = 0
                for step in range(args.n_epochs):
                    # training
                    loss_list = []
                    start = 0
                    # skip the last incomplete minibatch if its size < batch size
                    while start + args.batch_size <= train_data.shape[0]:
                        _, loss = model.train(sess, get_feed_dict(model, train_data, start, start + args.batch_size))
                        start += args.batch_size
                        loss_list.append(loss)
                    loss_mean = np.mean(loss_list)

                    train_nodea_emb_list, train_nodeb_emb_list, train_score, train_score_binary, train_auc, train_f1, train_aupr = ctr_eval(sess, args, model, train_data,args.batch_size)
                    eval_nodea_emb_list, eval_nodeb_emb_list, eval_score, eval_score_binary, eval_auc, eval_f1, eval_aupr = ctr_eval(sess, args, model, eval_data, args.batch_size)
                    test_nodea_emb_list, test_nodeb_emb_list, test_score, test_score_binary, test_auc, test_f1, test_aupr = ctr_eval(sess, args, model, test_data,args.batch_size)

                    # print('epoch %d    train auc: %.4f  f1: %.4f    train_aupr: %.4f    eval auc: %.4f  f1: %.4f    eval aupr: %.4f    test auc: %.4f  f1: %.4f  test_aupr: %.4f  loss: %.4f'
                    #     % (step, train_auc, train_f1, train_aupr, eval_auc, eval_f1,eval_aupr, test_auc, test_f1, test_aupr, loss_mean))

                    print("-"*50)
                    print('Epoch %d' % step + ':')
                    print('The AUC, AUPR and F1 values on the training data are: %.4f, %.4f, %.4f' %(train_auc, train_aupr, train_f1))
                    print('The AUC, AUPR and F1 values on the validation data are: %.4f, %.4f, %.4f' % (eval_auc, eval_aupr, eval_f1))
                    print('The AUC, AUPR and F1 values on the testing data are: %.4f, %.4f, %.4f' % (test_auc, test_aupr, test_f1))
                    print('The training loss is: %.4f' % loss_mean)

                    loss_curve.loc[step] = [step, loss_mean, train_auc, train_f1, train_aupr,eval_auc, eval_f1, eval_aupr, test_auc, test_f1,test_aupr]

                    # save the models with the highest eval_acu
                    if (eval_auc > best_eval_auc_flag):
                        best_eval_auc_flag = eval_auc
                        best_k = int(string[-1])
                        best_kk = kk
                        best_iteration = step

                        best_train_auc = train_auc
                        best_train_f1 = test_f1
                        best_train_aupr = train_aupr

                        best_eval_auc = eval_auc
                        best_eval_f1 = eval_f1
                        best_eval_aupr = eval_aupr

                        best_test_auc = test_auc
                        best_test_f1 = test_f1
                        best_test_aupr = test_aupr
                        best_loss = loss_mean

                        best_test_score = test_score
                        best_test_score_binary = test_score_binary

                        best_train_nodea_emb_list = train_nodea_emb_list
                        best_train_nodeb_emb_list = train_nodeb_emb_list
                        best_eval_nodea_emb_list = eval_nodea_emb_list
                        best_eval_nodeb_emb_list = eval_nodeb_emb_list
                        best_test_nodea_emb_list = test_nodea_emb_list
                        best_test_nodeb_emb_list = test_nodeb_emb_list

                        # saver.save(sess,'../best_models/best_model_' + str(best_k) + '_' + str(best_kk) + '.ckpt', global_step=best_iteration)




                    # early_stopping
                    if(args.earlystop_flag):
                        if (loss_mean < best_loss_flag):
                            stopping_step = 0
                            best_loss_flag = loss_mean
                        else:
                            stopping_step += 1
                            if (stopping_step >= early_stopping_flag):
                                print('Early stopping is trigger at step: %.4f  loss: %.4f  test_auc: %.4f  test_f1: %.4f   test_aupr: %.4f' % (step, loss_mean, test_auc, test_f1, test_aupr))
                                break

                # draw training curve
                loss_curve.to_csv('../results/loss_curve_' + string +'_' + str(kk) + '.csv', index=0)

                np.savetxt('../results/' + string + '_' + str(best_kk) + '_' + str(best_iteration) + '_train_nodea_emb.csv', (best_train_nodea_emb_list.eval())[args.batch_size:], delimiter='\t')
                np.savetxt('../results/' + string + '_' + str(best_kk) + '_' + str(best_iteration) + '_train_nodeb_emb.csv', (best_train_nodeb_emb_list.eval())[args.batch_size:], delimiter='\t')
                np.savetxt('../results/' + string + '_' + str(best_kk) + '_' + str(best_iteration) + '_eval_nodea_emb.csv', (best_eval_nodea_emb_list.eval())[args.batch_size:], delimiter='\t')
                np.savetxt('../results/' + string + '_' + str(best_kk) + '_' + str(best_iteration) + '_eval_nodeb_emb.csv', (best_eval_nodeb_emb_list.eval())[args.batch_size:], delimiter='\t')
                np.savetxt('../results/' + string + '_' + str(best_kk) + '_' + str(best_iteration) + '_test_nodea_emb.csv', (best_test_nodea_emb_list.eval())[args.batch_size:], delimiter='\t')
                np.savetxt('../results/' + string + '_' + str(best_kk) + '_' + str(best_iteration) + '_test_nodeb_emb.csv', (best_test_nodeb_emb_list.eval())[args.batch_size:], delimiter='\t')

                pd.DataFrame((np.array(best_test_score)).reshape(-1,1)).to_csv('../results/'+string+'_'+str(best_kk)+'_'+str(best_iteration)+'_scores.csv',header=False, index=False)
                pd.DataFrame((np.array(best_test_score_binary)).reshape(-1,1)).to_csv('../results/'+string+'_'+str(best_kk)+'_'+str(best_iteration)+'_scores_binary.csv',header=False, index=False)

                font = {
                    'family': 'SimHei',
                    'weight': 'normal',
                    'size': 20,
                    'color': 'black'
                }

                pl = plt.figure(figsize=(10, 10))
                # plt.suptitle('training curve', fontsize=25)
                plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace = 0.3, hspace = 0.3)
                plt.rcParams['savefig.dpi'] = 300
                plt.rcParams['figure.dpi'] = 300

                # add the first subplot
                pl.add_subplot(2, 2, 1)
                plt.plot(loss_curve['epoch'], loss_curve['loss'], linestyle='solid', color='#FF8C00', alpha=0.8, linewidth=3,label='loss')

                plt.legend(loc="upper right")
                plt.rcParams['legend.title_fontsize'] = 14
                plt.rcParams['legend.fontsize'] = 14
                plt.xlabel('Number Of Epochs', font)
                plt.ylabel('Training Loss', font)
                # plt.ylim(0.3, 0.9)
                plt.tick_params(labelsize=20)
                # plt.xticks([0, 1, 2, 3, 4, 5])
                # plt.xticks([0,5,10,15,20,25,30])
                # plt.title('loss_mean: ' + str(round(best_loss, 4)), font)

                # add the second subplot
                pl.add_subplot(2, 2, 2)
                plt.plot(loss_curve['epoch'], loss_curve['train_auc'], linestyle='dotted', color='#4169E1', alpha=0.8, linewidth=3,label='Train')
                plt.plot(loss_curve['epoch'], loss_curve['eval_auc'], linestyle='dashed', color='#FF4500', alpha=0.8, linewidth=3,label='Eval')
                plt.plot(loss_curve['epoch'], loss_curve['test_auc'], linestyle='dashdot', color='#228B22', alpha=0.8, linewidth=3,label='Test')

                plt.legend(loc="lower right")
                plt.rcParams['legend.title_fontsize'] = 14
                plt.rcParams['legend.fontsize'] = 14
                plt.xlabel('Number Of Epochs', font)
                plt.ylabel('AUC', font)
                # plt.ylim(0.84, 1.0)
                plt.tick_params(labelsize=20)
                # plt.xticks([0, 1, 2, 3, 4, 5])
                plt.xticks([0,5,10,15,20,25,30])
                # plt.title('test_auc: ' + str(round(best_test_auc, 4)), font)

                # add the third subplot
                pl.add_subplot(2, 2, 3)
                plt.plot(loss_curve['epoch'], loss_curve['train_f1'], linestyle='dotted', color='#4169E1', alpha=0.8, linewidth=3,label='Train',markerfacecolor='none')
                plt.plot(loss_curve['epoch'], loss_curve['eval_f1'], linestyle='dashed', color='#FF4500', alpha=0.8, linewidth=3,label='Eval',markerfacecolor='none')
                plt.plot(loss_curve['epoch'], loss_curve['test_f1'], linestyle='dashdot', color='#228B22', alpha=0.8, linewidth=3,label='Test',markerfacecolor='none')

                plt.legend(loc="lower right")
                plt.rcParams['legend.title_fontsize'] = 14
                plt.rcParams['legend.fontsize'] = 14
                plt.xlabel('Number Of Epochs', font)
                plt.ylabel('F1', font)
                # plt.ylim(0.84, 1.0)
                plt.tick_params(labelsize=20)
                # plt.xticks([0, 1, 2, 3, 4, 5])
                plt.xticks([0,5,10,15,20,25,30])
                # plt.title('test_f1: ' + str(round(best_test_f1, 4)), font)

                # add the fourth subplot
                pl.add_subplot(2, 2, 4)
                plt.plot(loss_curve['epoch'], loss_curve['train_aupr'], linestyle='dotted', color='#4169E1', alpha=0.8, linewidth=3,label='Train',markerfacecolor='none')
                plt.plot(loss_curve['epoch'], loss_curve['eval_aupr'], linestyle='dashed', color='#FF4500', alpha=0.8, linewidth=3,label='Eval',markerfacecolor='none')
                plt.plot(loss_curve['epoch'], loss_curve['test_aupr'], linestyle='dashdot', color='#228B22', alpha=0.8, linewidth=3,label='Test',markerfacecolor='none')

                plt.legend(loc="lower right")
                plt.rcParams['legend.title_fontsize'] = 14
                plt.rcParams['legend.fontsize'] = 14
                plt.xlabel('Number Of Epochs', font)
                plt.ylabel('AUPR', font)
                # plt.ylim(0.84, 1.0)
                plt.tick_params(labelsize=20)
                # plt.xticks([0, 1, 2, 3, 4, 5])
                plt.xticks([0,5,10,15,20,25,30])
                # plt.title('test_aupr: ' + str(round(best_test_aupr, 4)), font)

                # save curve
                pl.tight_layout()
                pl.savefig('../results/training_curve_' + string + '_' + str(kk)+'.png', bbox_inches='tight')
                pl.clf()

            tf.get_default_graph().finalize()

            kk=kk+1

            train_auc_kf_list.append(best_train_auc)
            train_f1_kf_list.append(best_train_f1)
            train_aupr_kf_list.append(best_train_aupr)

            eval_auc_kf_list.append(best_eval_auc)
            eval_f1_kf_list.append(best_eval_f1)
            eval_aupr_kf_list.append(best_eval_aupr)

            test_auc_kf_list.append(best_test_auc)
            test_f1_kf_list.append(best_test_f1)
            test_aupr_kf_list.append(best_test_aupr)

            loss_kf_list.append(best_loss)

            cross_validation = cross_validation + 1


    train_auc_kf_mean = np.mean(train_auc_kf_list)
    train_f1_kf_mean = np.mean(train_f1_kf_list)
    train_aupr_kf_mean = np.mean(train_aupr_kf_list)

    eval_auc_kf_mean = np.mean(eval_auc_kf_list)
    eval_f1_kf_mean = np.mean(eval_f1_kf_list)
    eval_aupr_kf_mean = np.mean(eval_aupr_kf_list)

    test_auc_kf_mean = np.mean(test_auc_kf_list)
    test_f1_kf_mean = np.mean(test_f1_kf_list)
    test_aupr_kf_mean = np.mean(test_aupr_kf_list)

    loss_kf_mean = np.mean(loss_kf_list)

    # print('train auc_std: %.4f  train f1_std: %.4f    train_aupr_std: %.4f    eval auc_std: %.4f  eval f1_std: %.4f eval_aupr_std: %.4f    test auc_std: %.4f  test f1_std: %.4f  test_aupr_std: %.4f  loss_std: %.4f'
    #     % (np.std(train_auc_kf_list), np.std(train_f1_kf_list), np.std(train_aupr_kf_list), np.std(eval_auc_kf_list), np.std(eval_f1_kf_list),
    #        np.std(eval_aupr_kf_list), np.std(test_auc_kf_list), np.std(test_f1_kf_list), np.std(test_aupr_kf_list), np.std(loss_kf_list)))

    print("-" * 50)
    print('final results')
    print('The std of AUC, AUPR and F1 values on the training data are: %.4f, %.4f, %.4f' % (np.std(train_auc_kf_list), np.std(train_aupr_kf_list), np.std(train_f1_kf_list)))
    print('The std of AUC, AUPR and F1 values on the validation data are: %.4f, %.4f, %.4f' % (np.std(eval_auc_kf_list), np.std(eval_aupr_kf_list), np.std(eval_f1_kf_list)))
    print('The std of AUC, AUPR and F1 values on the testing data are: %.4f, %.4f, %.4f' % (np.std(test_auc_kf_list), np.std(test_aupr_kf_list), np.std(test_f1_kf_list)))
    print('The std of training loss is: %.4f' % np.std(loss_kf_list))

    return loss_kf_mean, train_auc_kf_mean, train_f1_kf_mean, train_aupr_kf_mean, eval_auc_kf_mean, eval_f1_kf_mean, eval_aupr_kf_mean, test_auc_kf_mean, test_f1_kf_mean, test_aupr_kf_mean

def get_feed_dict(model, data, start, end):
    feed_dict = {model.nodea_indices: data[start:end, 0],
                 model.nodeb_indices: data[start:end, 1],
                 model.labels: data[start:end, 2]}
    return feed_dict


def ctr_eval(sess, args, model, data, batch_size):
    start = 0
    auc_list = []
    f1_list = []
    aupr_list = []
    scores_list = []
    scores_binary_list = []
    nodea_emb_list = tf.zeros([args.batch_size, args.dim])
    nodeb_emb_list = tf.zeros([args.batch_size, args.dim])

    while start + batch_size <= data.shape[0]:
        nodea_emb, nodeb_emb, scores_output, scores_binary_output, auc, f1, aupr = model.eval(sess, get_feed_dict(model, data, start, start + batch_size))

        nodea_emb_list = tf.concat([nodea_emb_list, nodea_emb], 0)
        nodeb_emb_list = tf.concat([nodeb_emb_list, nodeb_emb], 0)

        scores_list.append(scores_output)
        scores_binary_list.append(scores_binary_output)
        auc_list.append(auc)
        f1_list.append(f1)
        aupr_list.append(aupr)
        start += batch_size
    return nodea_emb_list, nodeb_emb_list, scores_list, scores_binary_list, float(np.mean(auc_list)), float(np.mean(f1_list)), float(np.mean(aupr_list))
