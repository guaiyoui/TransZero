import torch
from utils import f1_score_calculation, load_query_n_gt, cosin_similarity, dot_similarity, get_gt_legnth, coo_matrix_to_nx_graph
import argparse
import numpy as np
from WG_subgraph import mwg_subgraph_heuristic, mwg_subgraph_nocon
from tqdm import tqdm
import copy

def parse_args():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser()
    # main parameters
    parser.add_argument('--dataset', type=str, default='texas', help='dataset name')
    parser.add_argument('--EmbeddingPath', type=str, default='./pretrain_result/', help='embedding path')
    parser.add_argument('--iter_num', type=int, default=3, help='the number of iteration in the IWG.')

    return parser.parse_args()

def get_center_index(select_candidates, embedding_tensor):
    
    selected_embedding = embedding_tensor[select_candidates]

    pairwise_similarity = cosin_similarity(selected_embedding, selected_embedding)

    similarity_sum = torch.sum(pairwise_similarity, dim=0)
    # 可能可以有top2啥的操作？

    select_index = torch.argmin(similarity_sum)

    return select_candidates[select_index]


if __name__ == "__main__":
    args = parse_args()

    embedding_tensor = torch.from_numpy(np.load(args.EmbeddingPath + args.dataset + '.npy'))
    # embedding_tensor = torch.from_numpy(np.load(args.EmbeddingPath + args.dataset + '_subcon.npy'))
    
    # load queries and labels
    query, labels = load_query_n_gt("./dataset/", args.dataset, embedding_tensor.shape[0])
    
    revised_query = copy.deepcopy(query)
    original_query = copy.deepcopy(query)

    for iteration in range(args.iter_num):
        
        revised_query = copy.deepcopy(original_query)
        original_query = copy.deepcopy(query)

        query_feature = torch.mm(revised_query, embedding_tensor) # (query_num, embedding_dim)
        
        query_num = torch.sum(revised_query, dim=1)
        query_feature = torch.div(query_feature, query_num.view(-1, 1))
        
        # cosine similarity
        query_score = cosin_similarity(query_feature, embedding_tensor) # (query_num, node_num)

        y_pred = torch.zeros_like(query_score)
        for i in tqdm(range(query_score.shape[0])):
            query_index = (torch.nonzero(revised_query[i]).squeeze()).reshape(-1)
            
            selected_candidates, max_density = mwg_subgraph_nocon(query_index.tolist(), query_score[i].tolist()) 
            
            # 这边其实可以加一些点的label, 让模型往一个方向偏
            # if len(selected_candidates) == 0:
            #     print(query_index)

            selected_center = get_center_index(selected_candidates, embedding_tensor)
            original_query[i][selected_center] = -0.2

            
            # for j in query_index:
            #     if j != selected_center:
            #         query[i][j] = 0

            for j in range(len(selected_candidates)):
                y_pred[i][selected_candidates[j]] = 1

        
        f1_score = f1_score_calculation(y_pred, labels)

        print("F1 score by maximum weight gain: {:.4f}".format(f1_score))
    


