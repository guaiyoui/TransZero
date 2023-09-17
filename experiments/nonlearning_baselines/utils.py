import torch
from numpy import *

def f1_score_calculation(y_pred, y_true):
    if len(y_pred.shape) == 1:
        y_pred = y_pred.reshape(1, -1)
        y_true = y_true.reshape(1, -1)
    F1 = []
    for i in range(y_pred.shape[0]):
        pre = torch.sum(torch.multiply(y_pred[i], y_true[i]))/(torch.sum(y_pred[i])+1E-9)
        rec = torch.sum(torch.multiply(y_pred[i], y_true[i]))/(torch.sum(y_true[i])+1E-9)
        F1.append(2 * pre * rec / (pre + rec+1E-9))

    return mean(F1)


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