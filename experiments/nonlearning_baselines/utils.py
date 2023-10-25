import torch
from numpy import *
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, jaccard_score
import numpy as np

# def f1_score_calculation(y_pred, y_true):
#     if len(y_pred.shape) == 1:
#         y_pred = y_pred.reshape(1, -1)
#         y_true = y_true.reshape(1, -1)
#     F1 = []
#     for i in range(y_pred.shape[0]):
#         pre = torch.sum(torch.multiply(y_pred[i], y_true[i]))/(torch.sum(y_pred[i])+1E-9)
#         rec = torch.sum(torch.multiply(y_pred[i], y_true[i]))/(torch.sum(y_true[i])+1E-9)
#         F1.append(2 * pre * rec / (pre + rec+1E-9))

#     return mean(F1)

def f1_score_calculation(y_pred, y_true):
    y_pred = y_pred.reshape(1, -1)
    y_true = y_true.reshape(1, -1)
    pre = torch.sum(torch.multiply(y_pred, y_true))/(torch.sum(y_pred)+1E-9)
    rec = torch.sum(torch.multiply(y_pred, y_true))/(torch.sum(y_true)+1E-9)
    F1 = 2 * pre * rec / (pre + rec+1E-9)
    print("recall: ", rec, "pre: ", pre)
    return F1

def load_query_n_gt(path, dataset, vec_length):
    # load query and ground truth
    query = []
    file_query = open(path + dataset + '/' + dataset + ".query", 'r')
    for line in file_query:
        vec = [0 for i in range(vec_length)]
        line = line.strip()
        line = line.split(" ")
        line = [int(i) for i in line]
        # for i in line:
        #     vec[int(i)] = 1
        query.append(line)

    gt = []
    file_gt = open(path + dataset + '/' + dataset + ".gt", 'r')
    for line in file_gt:
        vec = [0 for i in range(vec_length)]
        line = line.strip()
        line = line.split(" ")
        
        for i in line:
            vec[int(i)] = 1
        gt.append(vec)
    
    return query, torch.Tensor(gt)

# def evaluation(comm_find, comm):

#     nmi_all, ari_all, jac_all = [], [], []

#     for i in range(comm_find.shape[0]):
#         nmi_all.append(NMI_score(comm_find[i], comm[i]))
#         ari_all.append(ARI_score(comm_find[i], comm[i]))
#         jac_all.append(JAC_score(comm_find[i], comm[i]))

#     return np.mean(nmi_all), np.mean(ari_all), np.mean(jac_all) 

def evaluation(comm_find, comm):
    comm_find = comm_find.reshape(-1)
    comm = comm.reshape(-1)

    return normalized_mutual_info_score(comm, comm_find), adjusted_rand_score(comm, comm_find), jaccard_score(comm, comm_find)


def NMI_score(comm_find, comm):

    score = normalized_mutual_info_score(comm, comm_find)
    #print("q, nmi:", score)
    return score

def ARI_score(comm_find, comm):

    score = adjusted_rand_score(comm, comm_find)
    #print("q, ari:", score)

    return score

def JAC_score(comm_find, comm):

    score = jaccard_score(comm, comm_find)
    #print("q, jac:", score)
    return score