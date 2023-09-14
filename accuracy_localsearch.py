import torch
from utils import f1_score_calculation, load_query_n_gt, cosin_similarity, dot_similarity, get_gt_legnth, coo_matrix_to_nx_graph_efficient
import argparse
import numpy as np
from WG_subgraph import mwg_subgraph_heuristic, mwg_subgraph_nocon
from tqdm import tqdm
from numpy import *

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

if __name__ == "__main__":
    args = parse_args()

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

    # load adj
    file_path = './dataset/'+args.dataset+'.pt'
    data_list = torch.load(file_path)
    adj = data_list[0]

    # graph = coo_matrix_to_nx_graph(adj)
    graph = coo_matrix_to_nx_graph_efficient(adj)
    # print(adj)

    print("query_score.shape: ", query_score.shape)
    # f1_score_collect = []

    y_pred = torch.zeros_like(query_score)
    for i in tqdm(range(query_score.shape[0])):
        query_index = (torch.nonzero(query[i]).squeeze()).reshape(-1)
        selected_candidates = mwg_subgraph_heuristic(query_index.tolist(), query_score[i].tolist(), graph)
        for j in range(len(selected_candidates)):
            y_pred[i][selected_candidates[j]] = 1
        
    f1_score = f1_score_calculation(y_pred.int(), labels.int())

    print("F1 score by maximum weight gain: {:.4f}".format(f1_score))


