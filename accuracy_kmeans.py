import torch
from utils import f1_score_calculation, load_query_n_gt, cosin_similarity, dot_similarity, get_gt_legnth
import argparse
import numpy as np
from kmeans_pytorch import kmeans
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
    parser.add_argument('--seed', type=int, default=0, help='the seed of model.')
    parser.add_argument('--num_communities', type=int, default=7, help='the number of communities.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    embedding_tensor = torch.from_numpy(np.load(args.EmbeddingPath + args.dataset + '.npy'))

    # load queries and labels
    query, labels = load_query_n_gt("./dataset/", args.dataset, embedding_tensor.shape[0])
    gt_length = get_gt_legnth("./dataset/", args.dataset)

    num_communities=args.num_communities
    # num_communities = 2

    communities = torch.zeros(num_communities, labels.shape[1]) 
    cluster_ids_x, cluster_centers = kmeans(X=embedding_tensor, num_clusters=num_communities, distance='cosine', device=torch.device('cuda:0'))
    
    for i in range(cluster_ids_x.shape[0]):
        communities[cluster_ids_x[i]][i] = 1
    
    F1_score = []
    for i in range(query.shape[0]):
        query_point = torch.nonzero(query[i])[0]
        F1_score.append(f1_score_calculation(communities[cluster_ids_x[query_point]].squeeze(), labels[i]))

    # print(F1_score)
    print("F1 score by kmeans: {:.4f}".format(mean(F1_score)))