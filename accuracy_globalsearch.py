import torch
from utils import f1_score_calculation, load_query_n_gt, cosin_similarity, dot_similarity, get_gt_legnth, coo_matrix_to_nx_graph_efficient
import argparse
import numpy as np
# from WG_subgraph import mwg_subgraph_heuristic, mwg_subgraph_nocon
from tqdm import tqdm
from numpy import *
import time

def parse_args():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser()
    # main parameters
    parser.add_argument('--dataset', type=str, default='cora', help='dataset name')
    parser.add_argument('--EmbeddingPath', type=str, default='./pretrain_result/', help='embedding path')
    parser.add_argument('--topk', type=int, default=400, help='the number of nodes selected.')

    return parser.parse_args()

def subgraph_density_controled(candidate_score, graph_score):
    
    weight_gain = (sum(candidate_score)-sum(graph_score)*(len(candidate_score)**1)/(len(graph_score)**1))/(len(candidate_score)**0.50)
    return weight_gain

def GlobalSearch(query_index, graph_score):

    # graph_score = MaxMinNormalization(graph_score, 0, 100)
    candidates = query_index
    selected_candidate = candidates

    start = time.time()
    graph_score=np.array(graph_score)
    max2min_index = np.argsort(-graph_score)
    end = time.time()
    # print("sort time: {:.4f}".format(end-start))
    max_density = -1000
    # candidate_score = [graph_score[i] for i in candidates]

    start = time.time()
    startpoint = 0
    endpoint = int(0.25*len(max2min_index))
    while True:
        candidates_half = query_index+[max2min_index[i] for i in range(0, int((startpoint+endpoint)/2))]
        candidate_score_half = [graph_score[i] for i in candidates_half]
        candidates_density_half = subgraph_density_controled(candidate_score_half, graph_score)

        candidates = query_index+[max2min_index[i] for i in range(0, endpoint)]
        candidate_score = [graph_score[i] for i in candidates]
        candidates_density = subgraph_density_controled(candidate_score, graph_score)

        if candidates_density>= candidates_density_half:
            startpoint = int((startpoint+endpoint)/2)
            endpoint = endpoint
        else:
            startpoint = startpoint
            endpoint = int((startpoint+endpoint)/2)
        # print(startpoint, endpoint)
        if startpoint == endpoint or startpoint+1 == endpoint:
            break

    selected_candidate = query_index+[max2min_index[i] for i in range(0, startpoint)] 
    
    end = time.time()
    # print("loop time: {:.4f}".format(end-start))
    return selected_candidate

if __name__ == "__main__":
    args = parse_args()

    if args.dataset == 'reddit' or args.dataset == 'aminer':
        embedding_tensor = torch.from_numpy(np.load(args.EmbeddingPath + args.dataset + '_2k.npy'))
    else:
        embedding_tensor = torch.from_numpy(np.load(args.EmbeddingPath + args.dataset + '.npy'))
    # embedding_tensor = torch.from_numpy(np.load(args.EmbeddingPath + args.dataset + '_subcon.npy'))
    
    # load queries and labels
    query, labels = load_query_n_gt("./dataset/", args.dataset, embedding_tensor.shape[0])
    # print(query[0:10], query.shape)
    gt_length = get_gt_legnth("./dataset/", args.dataset)

    query_feature = torch.mm(query, embedding_tensor) # (query_num, embedding_dim)
    
    query_num = torch.sum(query, dim=1)
    query_feature = torch.div(query_feature, query_num.view(-1, 1))
    
    # dot product plus softmax
    # query_score = dot_similarity(query_feature, embedding_tensor) # (query_num, node_num)

    # cosine similarity
    query_score = cosin_similarity(query_feature, embedding_tensor) # (query_num, node_num)
    query_score = torch.nn.functional.normalize(query_score, dim=1, p=1)

    # # load adj
    # file_path = './dataset/'+args.dataset+'.pt'
    # data_list = torch.load(file_path)
    # adj = data_list[0]

    # # graph = coo_matrix_to_nx_graph(adj)
    # graph = coo_matrix_to_nx_graph_efficient(adj)
    # print(adj)

    print("query_score.shape: ", query_score.shape)
    # f1_score_collect = []

    start = time.time()
    y_pred = torch.zeros_like(query_score)
    for i in tqdm(range(query_score.shape[0])):
        query_index = (torch.nonzero(query[i]).squeeze()).reshape(-1)

        selected_candidates = GlobalSearch(query_index.tolist(), query_score[i].tolist()) 
        for j in range(len(selected_candidates)):
            y_pred[i][selected_candidates[j]] = 1
        
        # print(len(selected_candidates))
        # f1_score = f1_score_calculation(y_pred[i].int(), labels[i].int())
        # print("F1 score by maximum weight gain: {:.4f}".format(f1_score))
    end = time.time()
    print("The global search using time: {:.4f}".format(end-start)) 
    f1_score = f1_score_calculation(y_pred.int(), labels.int())

    print("F1 score by maximum weight gain: {:.4f}".format(f1_score))


