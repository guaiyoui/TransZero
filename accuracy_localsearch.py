import torch
from utils import f1_score_calculation, load_query_n_gt, cosin_similarity, get_gt_legnth, coo_matrix_to_nx_graph_efficient, evaluation
import argparse
import numpy as np
from tqdm import tqdm
from numpy import *
import time
from utils import find_all_neighbors_bynx

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

def subgraph_density(candidate_score, avg_weight):
    weight_gain = (sum(candidate_score)-len(candidate_score)*avg_weight)/(len(candidate_score)**0.5)
    return weight_gain


def mwg_subgraph_heuristic(query_index, graph_score, graph):

    candidates = query_index

    selected_candidate = candidates
    max_density = -1000

    avg_weight = sum(graph_score)/len(graph_score)

    count = 0
    endpoint = int(0.25*len(graph_score))
    if endpoint >= 10000:
        endpoint = 10000
    
    while True:

        neighbors = find_all_neighbors_bynx(candidates, graph)
        
        if len(neighbors) == 0 or count>endpoint:
            break
        
        # select the index with the largest score.
        neighbor_score = [graph_score[i]for i in neighbors]
        i_index = neighbor_score.index(max(neighbor_score))
        
        candidates = candidates+[neighbors[i_index]]

        candidate_score = [graph_score[i]for i in candidates]
        candidates_density = subgraph_density(candidate_score, avg_weight)
        if candidates_density > max_density:
            max_density = candidates_density
            selected_candidate = candidates
        else:
            break

        count += 1
    
    return selected_candidate

if __name__ == "__main__":
    args = parse_args()
    print(args)
    embedding_tensor = torch.from_numpy(np.load(args.EmbeddingPath + args.dataset + '.npy'))
    
    # load queries and labels
    query, labels = load_query_n_gt("./dataset/", args.dataset, embedding_tensor.shape[0])
    gt_length = get_gt_legnth("./dataset/", args.dataset)

    # load adj
    if args.dataset in {"photo", "cs"}:
        file_path = './dataset/'+args.dataset+'_dgl.pt'
    else:
        file_path = './dataset/'+args.dataset+'_pyg.pt'
    data_list = torch.load(file_path)
    adj = data_list[0]

    graph = coo_matrix_to_nx_graph_efficient(adj)
    
    start = time.time()

    query_feature = torch.mm(query, embedding_tensor) # (query_num, embedding_dim)
    query_num = torch.sum(query, dim=1)
    query_feature = torch.div(query_feature, query_num.view(-1, 1))
    
    # cosine similarity
    query_score = cosin_similarity(query_feature, embedding_tensor) # (query_num, node_num)
    query_score = torch.nn.functional.normalize(query_score, dim=1, p=1)

    
    print("query_score.shape: ", query_score.shape)

    y_pred = torch.zeros_like(query_score)
    for i in tqdm(range(query_score.shape[0])):
        query_index = (torch.nonzero(query[i]).squeeze()).reshape(-1)
        selected_candidates = mwg_subgraph_heuristic(query_index.tolist(), query_score[i].tolist(), graph)
        for j in range(len(selected_candidates)):
            y_pred[i][selected_candidates[j]] = 1

    end = time.time()
    print("The local search using time: {:.4f}".format(end-start)) 
    print("The local search using time (one query): {:.4f}".format((end-start)/query_feature.shape[0])) 
    f1_score = f1_score_calculation(y_pred.int(), labels.int())
    
    print("F1 score by maximum weight gain (local search): {:.4f}".format(f1_score))

    nmi, ari, jac = evaluation(y_pred.int(), labels.int())
    
    print("NMI score by maximum weight gain (local search): {:.4f}".format(nmi))
    print("ARI score by maximum weight gain (local search): {:.4f}".format(ari))
    print("JAC score by maximum weight gain (local search): {:.4f}".format(jac))


