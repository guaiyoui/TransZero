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

def f1_score_calculation(y_pred, y_true):
    return f1_score(y_pred, y_true, average="macro")

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

def transform_csr_to_coo(adj):
    adj = adj.tocoo()
    adj = torch.sparse.LongTensor(torch.LongTensor([adj.row.tolist(), adj.col.tolist()]),
                              torch.LongTensor(adj.data.astype(np.int32)))
    return adj

def transform_sp_csr_to_coo(adj, batch_size, node_num):
    # chunks
    node_index = [i for i in range(node_num)]
    divide_index = [node_index[i:i+batch_size] for i in range(0, len(node_index), batch_size)]

    # adj of each chunks, in the format of sp_csr
    adj_sp_csr = [adj[divide_index[i]][:, divide_index[i]] for i in range(len(divide_index))]
    minus_adj_sp_csr = [sp.csr_matrix(torch.ones(item.shape))-item for item in adj_sp_csr]

    adj_tensor_coo = [transform_csr_to_coo(item).to_dense() for item in adj_sp_csr]
    minus_adj_tensor_coo = [transform_csr_to_coo(item).to_dense() for item in minus_adj_sp_csr]

    return adj_tensor_coo, minus_adj_tensor_coo


