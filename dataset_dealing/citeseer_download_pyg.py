# 导入相关库
import typing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

# 选择设备为GPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"device : {device}")

def edge_index_to_sparse_coo(edge_index):
    row = edge_index[0].long()
    col = edge_index[1].long()

    # 构建稀疏矩阵的形状
    num_nodes = torch.max(edge_index) + 1
    size = (num_nodes.item(), num_nodes.item())

    # 构建稀疏矩阵
    values = torch.ones_like(row)
    edge_index_sparse = torch.sparse_coo_tensor(torch.stack([row, col]), values, size)

    return edge_index_sparse

dataset_str = 'citeseer'

# 从PyG库加载Cora数据集
from torch_geometric.datasets import Planetoid
dataset = Planetoid(root='./data/citeseer', name='citeseer')
graph = dataset[0]

# print(graph.x, graph.edge_index, graph.y)
print(graph.edge_index)

torch.save([edge_index_to_sparse_coo(graph.edge_index).type(torch.LongTensor), graph.x.type(torch.LongTensor), graph.y.type(torch.LongTensor)], "./dataset/"+dataset_str+"_pyg.pt")

