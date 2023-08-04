import torch
from utils import f1_score_calculation, load_query_n_gt, cosin_similarity, dot_similarity
import argparse
import numpy as np


def parse_args():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser()
    # main parameters
    parser.add_argument('--dataset', type=str, default='cora', help='dataset name')
    parser.add_argument('--EmbeddingPath', type=str, default='./pretrain_result/', help='embedding path')
    parser.add_argument('--threshlod', type=float, default=0.5, help='threshold of the similarity')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    embedding_tensor = torch.from_numpy(np.load(args.EmbeddingPath + args.dataset + '.npy'))
    # embedding_tensor = torch.from_numpy(np.load(args.EmbeddingPath + args.dataset + '_subcon.npy'))
    
    # load queries and labels
    query, labels = load_query_n_gt("./dataset/", args.dataset, embedding_tensor.shape[0])

    query_feature = torch.mm(query, embedding_tensor) # (query_num, embedding_dim)
    
    query_num = torch.sum(query, dim=1)
    query_feature = torch.div(query_feature, query_num.view(-1, 1))
    
    # dot product plus softmax
    # query_score = dot_similarity(query_feature, embedding_tensor) # (query_num, node_num)

    # cosine similarity
    query_score = cosin_similarity(query_feature, embedding_tensor) # (query_num, node_num)

    print("query_score.shape: ", query_score.shape)
    print("query_score: ", query_score[0])

    y_pred = torch.where(query_score>= args.threshlod, 1, 0)

    f1_score = f1_score_calculation(y_pred, labels)

    print("F1 score by threshold: {:.4f}".format(f1_score))
