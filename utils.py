import argparse
import torch
import scipy.sparse as sp
from dgl import DGLGraph
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm
import dgl
from sklearn.metrics import f1_score
import scipy.sparse as sp
import numpy as np
import networkx as nx
from numpy import *

# Training settings
def parse_args():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser()

    # main parameters
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='cora',
                        help='Choose from {pubmed}')
    parser.add_argument('--device', type=int, default=2, 
                        help='Device cuda id')
    parser.add_argument('--seed', type=int, default=0, 
                        help='Random seed.')

    # model parameters
    parser.add_argument('--hops', type=int, default=7,
                        help='Hop of neighbors to be calculated')
    parser.add_argument('--pe_dim', type=int, default=15,
                        help='position embedding size')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden layer size')
    parser.add_argument('--ffn_dim', type=int, default=64,
                        help='FFN layer size')
    parser.add_argument('--n_layers', type=int, default=1,
                        help='Number of Transformer layers')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of Transformer heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout')
    parser.add_argument('--attention_dropout', type=float, default=0.1,
                        help='Dropout in the attention layer')
    parser.add_argument('--readout', type=str, default="mean")

    # training parameters
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='Batch size')
    parser.add_argument('--group_epoch_gap', type=int, default=20,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Number of epochs to train.')
    parser.add_argument('--tot_updates',  type=int, default=1000,
                        help='used for optimizer learning rate scheduling')
    parser.add_argument('--warmup_updates', type=int, default=400,
                        help='warmup steps')
    parser.add_argument('--peak_lr', type=float, default=0.001, 
                        help='learning rate')
    parser.add_argument('--end_lr', type=float, default=0.0001, 
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00001,
                        help='weight decay')
    parser.add_argument('--patience', type=int, default=50, 
                        help='Patience for early stopping')
    
    # model saving
    parser.add_argument('--save_path', type=str, default='./model/',
                        help='The path for the model to save')
    parser.add_argument('--model_name', type=str, default='cora',
                        help='The name for the model to save')
    parser.add_argument('--embedding_path', type=str, default='./pretrain_result/',
                        help='The path for the embedding to save')

    return parser.parse_args()


def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian

    #adjacency_matrix(transpose, scipy_fmt="csr")
    # A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    A = g.adj_external(scipy_fmt='csr')
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with scipy
    #EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2) # for 40 PEs
    EigVec = EigVec[:, EigVal.argsort()] # increasing order
    lap_pos_enc = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 

    return lap_pos_enc



def re_features(adj, features, K):
    #传播之后的特征矩阵,size= (N, 1, K+1, d )
    nodes_features = torch.empty(features.shape[0], 1, K+1, features.shape[1])

    for i in range(features.shape[0]):

        nodes_features[i, 0, 0, :] = features[i]

    x = features + torch.zeros_like(features)
    # print(adj, x)
    # print(type(adj), type(x))
    for i in range(K):

        x = torch.matmul(adj, x)

        for index in range(features.shape[0]):

            nodes_features[index, 0, i + 1, :] = x[index]        

    nodes_features = nodes_features.squeeze()


    return nodes_features

# def f1_score_calculation(y_pred, y_true):
#     return f1_score(y_pred, y_true, average="macro")
#     return f1_score(y_pred, y_true, average="binary")
#     return f1_score(y_pred, y_true)

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

    return F1

def load_query_n_gt(path, dataset, vec_length):
    # load query and ground truth
    query = []
    file_query = open(path + dataset + '/' + dataset + ".query", 'r')
    for line in file_query:
        vec = [0 for i in range(vec_length)]
        line = line.strip()
        line = line.split(" ")
        for i in line:
            vec[int(i)] = 1
        query.append(vec)

    gt = []
    file_gt = open(path + dataset + '/' + dataset + ".gt", 'r')
    for line in file_gt:
        vec = [0 for i in range(vec_length)]
        line = line.strip()
        line = line.split(" ")
        
        for i in line:
            vec[int(i)] = 1
        gt.append(vec)
    
    return torch.Tensor(query), torch.Tensor(gt)

def get_gt_legnth(path, dataset):
    gt_legnth = []
    file_gt = open(path + dataset + '/' + dataset + ".gt", 'r')
    for line in file_gt:
        line = line.strip()
        line = line.split(" ")
        gt_legnth.append(len(line))
    
    return torch.Tensor(gt_legnth)

def cosin_similarity(query_tensor, emb_tensor):
    similarity = torch.stack([torch.cosine_similarity(query_tensor[i], emb_tensor, dim=1) for i in range(len(query_tensor))], 0)
    return similarity
    
def dot_similarity(query_tensor, emb_tensor):
    similarity = torch.mm(query_tensor, emb_tensor.t()) # (query_num, node_num)
    similarity = torch.nn.Softmax(dim=1)(similarity)
    return similarity


def transform_coo_to_csr(adj):
    row=adj._indices()[0]
    col=adj._indices()[1]
    data=adj._values()
    shape=adj.size()
    adj=sp.csr_matrix((data, (row, col)), shape=shape)
    return adj

def transform_csr_to_coo(adj, size=None):
    adj = adj.tocoo()
    adj = torch.sparse.LongTensor(torch.LongTensor([adj.row.tolist(), adj.col.tolist()]),
                              torch.LongTensor(adj.data.astype(np.int32)),
                              torch.Size([size, size]))
    return adj

def transform_sp_csr_to_coo(adj, batch_size, node_num):
    # chunks
    node_index = [i for i in range(node_num)]
    divide_index = [node_index[i:i+batch_size] for i in range(0, len(node_index), batch_size)]

    # adj of each chunks, in the format of sp_csr
    print("start mini batch: adj of each chunks")
    adj_sp_csr = [adj[divide_index[i]][:, divide_index[i]] for i in range(len(divide_index))]
    print("start mini batch: minus adj of each chunks")
    minus_adj_sp_csr = [sp.csr_matrix(torch.ones(item.shape))-item for item in adj_sp_csr]

    # adj_tensor_coo = [transform_csr_to_coo(item).to_dense() for item in adj_sp_csr]
    # minus_adj_tensor_coo = [transform_csr_to_coo(item).to_dense() for item in minus_adj_sp_csr]
    print("start mini batch: back to torch coo adj")
    adj_tensor_coo = [transform_csr_to_coo(adj_sp_csr[i], len(divide_index[i])).to_dense() for i in range(len(divide_index))]
    print("start mini batch: back to torch coo minus adj")
    minus_adj_tensor_coo = [transform_csr_to_coo(minus_adj_sp_csr[i], len(divide_index[i])).to_dense() for i in range(len(divide_index))]

    return adj_tensor_coo, minus_adj_tensor_coo


# transform coo to edge index in pytorch geometric 
def transform_coo_to_edge_index(adj):
    adj = adj.coalesce()
    edge_index = adj.indices().detach().long()
    return edge_index

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# determine one edge in edge_index or not of torch geometric
def is_edge_in_edge_index(edge_index, source, target):
    mask = (edge_index[0] == source) & (edge_index[1] == target)
    return mask.any()

def construct_pseudo_assignment(cluster_ids_x):
    pseudo_assignment = torch.zeros(cluster_ids_x.shape[0], int(cluster_ids_x.max()+1))

    for i in range(cluster_ids_x.shape[0]):
        pseudo_assignment[i][int(cluster_ids_x[i])] = 1
    
    return pseudo_assignment

def pq_computation(similarity):
    q = torch.nn.functional.normalize(similarity, dim=1, p=1)
    p_temp = torch.mul(q, q)
    q_colsum = torch.sum(q, axis=0)
    p_temp = torch.div(p_temp,q_colsum)
    p = torch.nn.functional.normalize(p_temp, dim=1, p=1)
    return q, p

def coo_matrix_to_nx_graph(matrix):
    # Create an empty NetworkX graph
    graph = nx.Graph()

    # Get the number of nodes in the COO matrix
    num_nodes = matrix.shape[0]

    # Convert the COO matrix to a dense matrix
    dense_matrix = matrix.to_dense()
    # graph.add_edge(0, 105)
    # graph.add_edge(0, 120)
    # graph.add_edge(0, 176)
    # print(dense_matrix[120][120:125])
    # Iterate over the non-zero entries in the dense matrix
    for i in range(num_nodes):
        for j in range(num_nodes):
            if dense_matrix[i][j] != 0:
                # Add an edge to the NetworkX graph
                graph.add_edge(i, j)
                graph.add_edge(j, i)

    return graph

def coo_matrix_to_nx_graph_efficient(adj_matrix):
    # 创建一个无向图对象
    graph = nx.Graph()

    # 获取 COO 矩阵的行和列索引以及权重值
    adj_matrix = adj_matrix.coalesce()
    rows = adj_matrix.indices()[0]
    cols = adj_matrix.indices()[1]
    # print(rows, cols)
    # 添加节点和边到图中
    for i in range(len(rows)):
        graph.add_edge(int(rows[i]), int(cols[i]))
        graph.add_edge(int(cols[i]), int(rows[i]))

    return graph

def obtain_adj_from_nx(graph):
    return np.array(nx.adjacency_matrix(graph, nodelist=[i for i in range(max(graph.nodes)+1)]).todense())

def find_all_neighbors_bynx(query, Graph):
    neighbors = []
    for i in range(len(query)):
        for j in Graph.neighbors(query[i]):
            if j not in query:
                neighbors.append(j)
    return neighbors

def MaxMinNormalization(x, Min, Max):
    
    x = np.array(x)
    x_max = np.max(x)
    x_min = np.min(x)

    x = [(item-x_min)*(Max-Min)/(x_max - x_min) + Min for item in x]

    return x