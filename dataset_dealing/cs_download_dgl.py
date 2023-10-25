import dgl.data
import dgl
import scipy.sparse as sp
import networkx as nx
import torch
# 下载数据集
dataset = dgl.data.CoauthorCSDataset()

# 获取图和标签
graph = dataset[0]
labels = graph.ndata['label']

# 获取节点特征
node_features = graph.ndata['feat']
print(node_features, node_features.shape)
print(node_features[0][0:20])
node_features = torch.sign(node_features)
print(node_features[0][0:20])
# node_features = node_features.type(torch.LongTensor)
# print(node_features[0][0:20])

# 获取节点标签
node_labels = graph.ndata['label']

edges = graph.edges()

# 将图转换为邻接矩阵, 用sparse.csr存
# nx_graph = graph.to_networkx()
# adj = nx.linalg.graphmatrix.adjacency_matrix(nx_graph)

# 将图转换为邻接矩阵, 用sparse.csr存
# 将邻接矩阵转换为 COO 格式的稀疏张量

i = [edges[0].tolist(), edges[1].tolist()]
v = [1 for j in range(len(edges[0].tolist()))]
adj = torch.sparse.LongTensor(torch.LongTensor(i),
                              torch.LongTensor(v),
                              torch.Size([(node_labels.shape)[0], (node_labels.shape)[0]]))

def edge_index_to_sparse_coo(edge_index):
    row = edge_index[0].long()
    col = edge_index[1].long()

    # 构建稀疏矩阵的形状
    num_nodes = torch.max(edge_index) + 1
    size = (num_nodes.item(), num_nodes.item())
    size = (18333, 18333)
    # 构建稀疏矩阵
    values = torch.ones_like(row)
    edge_index_sparse = torch.sparse_coo_tensor(torch.stack([row, col]), values, size)

    return edge_index_sparse

# print(adj)
# print(edges[0].tolist())
edge0 = edges[0].tolist() 
edge0 = [int(i) for i in edge0]
edge1 = edges[1].tolist() 
edge1 = [int(i) for i in edge1]
edges = torch.Tensor([edge0, edge1]).type(torch.int)
print(edges)
adj = edge_index_to_sparse_coo(edges)

torch.save([adj.type(torch.LongTensor), node_features, node_labels.type(torch.LongTensor)], "./dataset/cs_dgl.pt")


