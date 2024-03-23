import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from operator import itemgetter
from scipy.stats import entropy
from math import log
from sklearn.metrics import roc_auc_score

top1 = 1
top2 = 5
top3 = 10
top4 = 15


# calculate NDCG@k
def NDCG_at_k(predicted_list, ground_truth, k):
    dcg_value = [(v / log(i + 1 + 1, 2)) for i, v in enumerate(predicted_list[:k])]
    dcg = np.sum(dcg_value)
    if len(ground_truth) < k:
        ground_truth += [0 for i in range(k - len(ground_truth))]
    idcg_value = [(v / log(i + 1 + 1, 2)) for i, v in enumerate(ground_truth[:k])]
    idcg = np.sum(idcg_value)
    return dcg / idcg


# calculate precision@k, recall@k, NDCG@k, where k = 1,5,10,15
def user_precision_recall_ndcg(new_user_prediction, test):
    dcg_list = []

    # compute the number of true positive items at top k
    count_1, count_5, count_10, count_15 = 0, 0, 0, 0
    for i in range(15):
        if i == 0 and new_user_prediction[i][0] in test:
            count_1 = 1.0
        if i < 5 and new_user_prediction[i][0] in test:
            count_5 += 1.0
        if i < 10 and new_user_prediction[i][0] in test:
            count_10 += 1.0
        if new_user_prediction[i][0] in test:
            count_15 += 1.0
            dcg_list.append(1)
        else:
            dcg_list.append(0)

    # calculate NDCG@k
    idcg_list = [1 for i in range(len(test))]
    ndcg_tmp_1 = NDCG_at_k(dcg_list, idcg_list, 1)
    ndcg_tmp_5 = NDCG_at_k(dcg_list, idcg_list, 5)
    ndcg_tmp_10 = NDCG_at_k(dcg_list, idcg_list, 10)
    ndcg_tmp_15 = NDCG_at_k(dcg_list, idcg_list, 15)

    # precision@k
    precision_1 = count_1
    precision_5 = count_5 / 5.0
    precision_10 = count_10 / 10.0
    precision_15 = count_15 / 15.0

    l = len(test)
    if l == 0:
        l = 1
    # recall@k
    recall_1 = count_1 / l
    recall_5 = count_5 / l
    recall_10 = count_10 / l
    recall_15 = count_15 / l

    # return precision, recall, ndcg_tmp
    return np.array([precision_1, precision_5, precision_10, precision_15]), \
           np.array([recall_1, recall_5, recall_10, recall_15]), \
           np.array([ndcg_tmp_1, ndcg_tmp_5, ndcg_tmp_10, ndcg_tmp_15])


# calculate precision@k, recall@k, NDCG@k, where k = 1,5,10,15

def auc_per_user(Rec, test_df, train_df):
    Rec = copy.copy(Rec)
    
    user_num = Rec.shape[0]
    item_num = Rec.shape[1]
    items = list(range(item_num))
    scores = []
    
    for u in range(user_num):
        like_item = (train_df.loc[train_df['user_id'] == u, 'item_id']).tolist()
        Rec[u, like_item] = 0

    for u in range(user_num):  # iterate each user
        u_test = (test_df.loc[test_df['user_id'] == u, 'item_id']).tolist()
        u_pred = Rec[u, :].reshape(-1)
        
        grnd = np.zeros(item_num, dtype=np.int32)
        for p in u_test:
            index = items.index(p)
            grnd[index] = 1
        scores.append(roc_auc_score(grnd, u_pred))

    return sum(scores) / len(scores)

def auc_per_user_multiclass(Rec, test_df, train_df):
    Rec = copy.copy(Rec)
    
    user_num = Rec.shape[0]
    item_num = Rec.shape[1]
    items = list(range(item_num))
    scores = []
    
    for u in range(user_num):
        like_item = (train_df.loc[train_df['user_id'] == u, 'item_id']).tolist()
        Rec[u, like_item] = 0

    for u in range(user_num):  # iterate each user
        u_test = (test_df.loc[test_df['user_id'] == u, 'item_id']).tolist()
        u_pred = Rec[u, :].reshape(-1)
        
        grnd = np.zeros(item_num, dtype=np.int32)
        for p in u_test:
            index = items.index(p)
            grnd[index] = 1
        scores.append(roc_auc_score(grnd, u_pred, multi_class='ovr'))

    return sum(scores) / len(scores)

def auc_per_user_type(Rec, test_df, train_df, user_test, user_idd_type_list, key_type):
    test_dict = dict()
    user_id_type_test_dict = dict()
    auc = dict()   

    for k in key_type:
        test_dict[k] = 0.0
        user_id_type_test_dict[k] = []
        
    for t in user_test:
        gl = user_idd_type_list[t]
        for g in gl:
            if g in key_type:
                test_dict[g] += 1.0
                user_id_type_test_dict[g].append(t)
                
    for g in key_type:
        auc[g] = 0.0

    Rec = copy.copy(Rec)
    
    user_num = Rec.shape[0]
    item_num = Rec.shape[1]
    items = list(range(item_num))
    scores = []
   
    for u in range(user_num):
        like_item = (train_df.loc[train_df['user_id'] == u, 'item_id']).tolist()
        Rec[u, like_item] = 0
        
    for g in key_type:
        auc_per_type = []
 
        user_id_per_type = user_id_type_test_dict[g]
        for u in user_id_per_type:  # iterate each user in each user type
            u_test = (test_df.loc[test_df['user_id'] == u, 'item_id']).tolist()
            u_pred = Rec[u, :].reshape(-1)

            grnd = np.zeros(item_num, dtype=np.int32)
            for p in u_test:
                index = items.index(p)
                grnd[index] = 1
            auc_per_type.append(roc_auc_score(grnd, u_pred))

        auc[g] = sum(auc_per_type) / len(auc_per_type)
 
    return auc

def auc_per_user_type_multiclass(Rec, test_df, train_df, user_test, user_idd_type_list, key_type):
    test_dict = dict()
    user_id_type_test_dict = dict()
    auc = dict()   

    for k in key_type:
        test_dict[k] = 0.0
        user_id_type_test_dict[k] = []
        
    for t in user_test:
        gl = user_idd_type_list[t]
        for g in gl:
            if g in key_type:
                test_dict[g] += 1.0
                user_id_type_test_dict[g].append(t)
                
    for g in key_type:
        auc[g] = 0.0

    Rec = copy.copy(Rec)
    
    user_num = Rec.shape[0]
    item_num = Rec.shape[1]
    items = list(range(item_num))
    scores = []
   
    for u in range(user_num):
        like_item = (train_df.loc[train_df['user_id'] == u, 'item_id']).tolist()
        Rec[u, like_item] = 0
        
    for g in key_type:
        auc_per_type = []
 
        user_id_per_type = user_id_type_test_dict[g]
        for u in user_id_per_type:  # iterate each user in each user type
            u_test = (test_df.loc[test_df['user_id'] == u, 'item_id']).tolist()
            u_pred = Rec[u, :].reshape(-1)

            grnd = np.zeros(item_num, dtype=np.int32)
            for p in u_test:
                index = items.index(p)
                grnd[index] = 1
            auc_per_type.append(roc_auc_score(grnd, u_pred, multi_class='ovr'))

        auc[g] = sum(auc_per_type) / len(auc_per_type)
 
    return auc

    
def metric_per_user_type(Rec, test_df, train_df, user_test, user_idd_type_list, key_type):
    test_dict = dict()
    user_id_type_test_dict = dict()
   

    for k in key_type:
        test_dict[k] = 0.0
        user_id_type_test_dict[k] = []
        
    for t in user_test:
        gl = user_idd_type_list[t]
        for g in gl:
            if g in key_type:
                test_dict[g] += 1.0
                user_id_type_test_dict[g].append(t)


    precision = dict()
    recall = dict()
    ndcg = dict()
   
    
    for g in key_type:
        precision[g] = []
        recall[g] = []
        ndcg[g] = []
    
    user_num = Rec.shape[0]

    for i in range(user_num):
        like_item = (train_df.loc[train_df['user_id'] == i, 'item_id']).tolist()
        Rec[i, like_item] = -100000.0

    for g in key_type:
        precision_per_type = np.array([0.0, 0.0, 0.0, 0.0])
        recall_per_type = np.array([0.0, 0.0, 0.0, 0.0])
        ndcg_per_type = np.array([0.0, 0.0, 0.0, 0.0])
        
        user_id_per_type = user_id_type_test_dict[g]
        for u in user_id_per_type:  # iterate each user in each user type
            u_test = (test_df.loc[test_df['user_id'] == u, 'item_id']).tolist()
            u_pred = Rec[u, :]

            top15_item_idx_no_train = np.argpartition(u_pred, -15)[-15:]
            top15 = (np.array([top15_item_idx_no_train, u_pred[top15_item_idx_no_train]])).T
            top15 = sorted(top15, key=itemgetter(1), reverse=True)

            # calculate the metrics
            num_user_per_type = len(user_id_per_type)
            if not len(u_test) == 0:
                precision_u, recall_u, ndcg_u = user_precision_recall_ndcg(top15, u_test)
                precision_per_type += precision_u
                recall_per_type += recall_u
                ndcg_per_type += ndcg_u
            else:
                num_user_per_type -= 1

        # compute the average over all users per type
        precision_per_type /= len(user_id_per_type)
        recall_per_type /= len(user_id_per_type)
        ndcg_per_type /= len(user_id_per_type)
        
#        print('precision per type: ',precision_per_type)
        precision[g].append(precision_per_type)
#        print('precision per type g: ',precision[g])
        recall[g].append(recall_per_type)
        ndcg[g].append(ndcg_per_type)
#    print('user call ',precision['M'])
    return precision, recall, ndcg

# calculate the metrics of the result
def test_model_all(Rec, test_df, train_df):
    Rec = copy.copy(Rec)
    precision = np.array([0.0, 0.0, 0.0, 0.0])
    recall = np.array([0.0, 0.0, 0.0, 0.0])
    ndcg = np.array([0.0, 0.0, 0.0, 0.0])
    user_num = Rec.shape[0]

    for i in range(user_num):
        like_item = (train_df.loc[train_df['user_id'] == i, 'item_id']).tolist()
        Rec[i, like_item] = -100000.0

    for u in range(user_num):  # iterate each user
        u_test = (test_df.loc[test_df['user_id'] == u, 'item_id']).tolist()
        u_pred = Rec[u, :]

        top15_item_idx_no_train = np.argpartition(u_pred, -15)[-15:]
        top15 = (np.array([top15_item_idx_no_train, u_pred[top15_item_idx_no_train]])).T
        top15 = sorted(top15, key=itemgetter(1), reverse=True)

        # calculate the metrics
        if not len(u_test) == 0:
            precision_u, recall_u, ndcg_u = user_precision_recall_ndcg(top15, u_test)
            precision += precision_u
            recall += recall_u
            ndcg += ndcg_u
        else:
            user_num -= 1

    # compute the average over all users
    precision /= user_num
    recall /= user_num
    ndcg /= user_num
    print('precision_1\t[%.7f],\t||\t precision_5\t[%.7f],\t||\t precision_10\t[%.7f],\t||\t precision_15\t[%.7f]' \
          % (precision[0], precision[1], precision[2], precision[3]))
    print('recall_1   \t[%.7f],\t||\t recall_5   \t[%.7f],\t||\t recall_10   \t[%.7f],\t||\t recall_15   \t[%.7f]' \
          % (recall[0], recall[1], recall[2], recall[3]))
    f_measure_1 = 2 * (precision[0] * recall[0]) / (precision[0] + recall[0]) if not precision[0] + recall[
        0] == 0 else 0
    f_measure_5 = 2 * (precision[1] * recall[1]) / (precision[1] + recall[1]) if not precision[1] + recall[
        1] == 0 else 0
    f_measure_10 = 2 * (precision[2] * recall[2]) / (precision[2] + recall[2]) if not precision[2] + recall[
        2] == 0 else 0
    f_measure_15 = 2 * (precision[3] * recall[3]) / (precision[3] + recall[3]) if not precision[3] + recall[
        3] == 0 else 0
    print('f_measure_1\t[%.7f],\t||\t f_measure_5\t[%.7f],\t||\t f_measure_10\t[%.7f],\t||\t f_measure_15\t[%.7f]' \
          % (f_measure_1, f_measure_5, f_measure_10, f_measure_15))
    f_score = [f_measure_1, f_measure_5, f_measure_10, f_measure_15]
    print('ndcg_1     \t[%.7f],\t||\t ndcg_5     \t[%.7f],\t||\t ndcg_10     \t[%.7f],\t||\t ndcg_15     \t[%.7f]' \
          % (ndcg[0], ndcg[1], ndcg[2], ndcg[3]))
    return precision, recall, f_score, ndcg

#calcul de metrique pour chaque type d'utilisateur
def test_model_per_user_type(Rec, test_df, train_df, user_idd_type_list, key_type):
    Rec = copy.copy(Rec)
    
    precision = dict()
    recall = dict()
    ndcg = dict()
    auc = dict()
    
    for k in key_type:
        precision[k] = np.array([0.0, 0.0, 0.0, 0.0])
        recall[k] = np.array([0.0, 0.0, 0.0, 0.0])
        ndcg[k] = np.array([0.0, 0.0, 0.0, 0.0])
    
    user_num = Rec.shape[0]
    user_test = list(range(user_num))
    precision, recall, ndcg = metric_per_user_type(Rec, test_df, train_df, user_test, user_idd_type_list, key_type)
    auc = auc_per_user_type(Rec, test_df, train_df, user_test, user_idd_type_list, key_type)
#    print('test precision',precision)
    for k in key_type:
        print('Metrics for user type\t',k)
#         print('test precision',precision)
        print('precision_1\t[%.7f],\t||\t precision_5\t[%.7f],\t||\t precision_10\t[%.7f],\t||\t precision_15\t[%.7f]' \
          % (precision[k][0].tolist()[0], precision[k][0].tolist()[1], precision[k][0].tolist()[2], precision[k][0].tolist()[3]))
        print('recall_1\t[%.7f],\t||\t recall_5\t[%.7f],\t||\t recall_10\t[%.7f],\t||\t recall_15\t[%.7f]' \
          % (recall[k][0].tolist()[0], recall[k][0].tolist()[1], recall[k][0].tolist()[2], recall[k][0].tolist()[3]))
        print('ndcg_1\t[%.7f],\t||\t ndcg_5\t[%.7f],\t||\t ndcg_10\t[%.7f],\t||\t ndcg_15\t[%.7f]' \
          % (ndcg[k][0].tolist()[0], ndcg[k][0].tolist()[1], ndcg[k][0].tolist()[2], ndcg[k][0].tolist()[3]))
        print('AUC per user type\t[%.7f]' % (auc[k]))
    return precision, recall, ndcg, auc

#calcul de metrique pour chaque type d'utilisateur
def test_model_per_user_type_multiclass(Rec, test_df, train_df, user_idd_type_list, key_type):
    Rec = copy.copy(Rec)
    
    precision = dict()
    recall = dict()
    ndcg = dict()
    auc = dict()
    
    for k in key_type:
        precision[k] = np.array([0.0, 0.0, 0.0, 0.0])
        recall[k] = np.array([0.0, 0.0, 0.0, 0.0])
        ndcg[k] = np.array([0.0, 0.0, 0.0, 0.0])
    
    user_num = Rec.shape[0]
    user_test = list(range(user_num))
    precision, recall, ndcg = metric_per_user_type(Rec, test_df, train_df, user_test, user_idd_type_list, key_type)
    auc = auc_per_user_type_multiclass(Rec, test_df, train_df, user_test, user_idd_type_list, key_type)
#    print('test precision',precision)
    for k in key_type:
        print('Metrics for user type\t',k)
#         print('test precision',precision)
        print('precision_1\t[%.7f],\t||\t precision_5\t[%.7f],\t||\t precision_10\t[%.7f],\t||\t precision_15\t[%.7f]' \
          % (precision[k][0].tolist()[0], precision[k][0].tolist()[1], precision[k][0].tolist()[2], precision[k][0].tolist()[3]))
        print('recall_1\t[%.7f],\t||\t recall_5\t[%.7f],\t||\t recall_10\t[%.7f],\t||\t recall_15\t[%.7f]' \
          % (recall[k][0].tolist()[0], recall[k][0].tolist()[1], recall[k][0].tolist()[2], recall[k][0].tolist()[3]))
        print('ndcg_1\t[%.7f],\t||\t ndcg_5\t[%.7f],\t||\t ndcg_10\t[%.7f],\t||\t ndcg_15\t[%.7f]' \
          % (ndcg[k][0].tolist()[0], ndcg[k][0].tolist()[1], ndcg[k][0].tolist()[2], ndcg[k][0].tolist()[3]))
        print('AUC per user type\t[%.7f]' % (auc[k]))
    return precision, recall, ndcg, auc

def negative_sample(train_df, num_user, num_item, neg):
    user = []
    item_pos = []
    item_neg = []
    item_set = set(range(num_item))
    for i in range(num_user):
        like_item = (train_df.loc[train_df['user_id'] == i, 'item_id']).tolist()
        unlike_item = list(item_set - set(like_item))
        if len(unlike_item) < neg:
            tmp_neg = len(unlike_item)
        else:
            tmp_neg = neg
        for l in like_item:
            neg_samples = (np.random.choice(unlike_item, size=tmp_neg, replace=False)).tolist()
            user += [i] * tmp_neg
            item_pos += [l] * tmp_neg
            item_neg += neg_samples
    num_sample = len(user)
    return num_sample, np.array(user).reshape((num_sample, 1)),\
           np.array(item_pos).reshape((num_sample, 1)), np.array(item_neg).reshape((num_sample, 1))



def relative_std(dictionary):
    tmp = []
    for key, value in sorted(dictionary.items(), key=lambda item: (item[1], item[0])):
        tmp.append(value)
    rstd = np.std(tmp) / (np.mean(tmp) + 1e-10)
    return rstd
