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

    return parser.parse_args()


def subgraph_density_controled(candidate_score, graph_score, node_num):
    weight_gain = (torch.sum(candidate_score, dim=1)-torch.sum(graph_score, dim=1)*(node_num**1)/(graph_score.shape[1]))/(node_num**0.30)
    return weight_gain

if __name__ == "__main__":

    args = parse_args()

    embedding_tensor = torch.from_numpy(np.load(args.EmbeddingPath + args.dataset + '.npy'))

    # load queries and labels
    query, labels = load_query_n_gt("./dataset/", args.dataset, embedding_tensor.shape[0])

    print(query.shape, labels.shape, embedding_tensor.shape)

    y_pred = torch.zeros_like(labels)
    for i in tqdm(range(query.shape[0])):
        
        candidates = query[i].view(1, -1)
        max_density = -1000

        while True:
            query_feature = torch.mm(candidates, embedding_tensor) # (query_num, embedding_dim)
            query_num = torch.sum(candidates, dim=1)
            
            if query_num >= int(0.25*embedding_tensor.shape[0]):
                print(query_num)
                break
            query_feature = torch.div(query_feature, query_num.view(-1, 1))
            
            # cosine similarity
            query_score = cosin_similarity(query_feature, embedding_tensor) # (query_num, node_num)
            query_score = torch.nn.functional.normalize(query_score, dim=1, p=1)
            score_notinselected = torch.multiply(query_score, 1-candidates)
            
            candidates[0, torch.argmax(score_notinselected)] = 1

            candidates_score = torch.multiply(query_score, candidates)
            # print(candidates_score, query_score)

            candidates_density = subgraph_density_controled(candidates_score, query_score, node_num = query_num+1)
            if candidates_density > max_density:
                max_density = candidates_density
                # selected_candidate = candidates
            else:
                print(query_num)
                break

        y_pred[i] = candidates

        f1_score = f1_score_calculation(y_pred[i].int(), labels[i].int())

        print("F1 score by maximum weight gain: {:.4f}".format(f1_score))
    
    f1_score = f1_score_calculation(y_pred.int(), labels.int())

    print("F1 score by maximum weight gain: {:.4f}".format(f1_score))

            
            

        
