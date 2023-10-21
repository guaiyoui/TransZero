import numpy as np
import os.path as osp
import os
import sys
import warnings
import torch
warnings.filterwarnings('ignore')
from torch_sparse import coalesce
import scipy.sparse as sp
from torch_geometric.datasets import Reddit

reddit = Reddit(root='../../../COCLEP/COCLEP/dataset/reddit/')

reddit = reddit[0]
label_reddit = reddit.y.numpy()
edge_index = reddit.edge_index

unique_classed_reddit, class_counts_reddit = np.unique(label_reddit, return_counts=True)
print('Reddit dataset label:', unique_classed_reddit,"\t count:" ,class_counts_reddit)


labels = label_reddit.reshape(-1)


# row = torch.tensor(edge_index[0])
# col = torch.tensor(edge_index[1])
# value = torch.ones(row.shape[0])

# adj = torch.sparse_coo_tensor(
#     indices=torch.stack([row, col], dim=0),
#     values=value,
#     size=(label_reddit.shape[0], label_reddit.shape[0])
# )

data = np.load('../../../NAGphormer/dataset/reddit_adj.npz')
# 从data中获取相关属性
indices = data['indices']
indptr = data['indptr']
shape = data['shape']
data = data['data']

# 创建稀疏矩阵对象
adj = sp.csr_matrix((data, indices, indptr), shape=shape, dtype=float)

features = reddit.x

print(adj, type(adj))
print(features, features.shape)
torch.save([adj, features.type(torch.LongTensor), reddit.y.type(torch.LongTensor)], "../reddit.pt")