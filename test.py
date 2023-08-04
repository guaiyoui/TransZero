# from data_loader import get_dataset

# import time
# import utils
# import random
# import argparse
# import numpy as np
# import torch
# import torch.nn.functional as F
# from early_stop import EarlyStopping, Stop_args
# from model import PretrainModel
# from lr import PolynomialDecayLR
# import os.path
# import torch.utils.data as Data
# from utils import *



# if __name__ == "__main__":

#     args = parse_args()
#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(args.seed)
    
#     adj, features, labels, idx_train, idx_val, idx_test = get_dataset(args.dataset, args.pe_dim)

#     print(adj)

import torch
import scipy.sparse as sp
from utils import *
import numpy as np


a = torch.Tensor([[1, 2],
                  [3, 0]])

b = torch.Tensor([[1, 1],
                  [1, 0]])

c = b.to_sparse_coo()
print(c)
print(transform_coo_to_csr(c))

# a = sp.csr_matrix(a)
# b = sp.csr_matrix(b)
# print(a-b)


node_index = [i for i in range(5)]
divide_index = [node_index[i:i+3] for i in range(0, len(node_index), 3)]
print(divide_index)
d = torch.Tensor([[1, 1, 1, 2, 3],
                  [1, 0, 0, 1, 2],
                  [2, 0, 1, 1, 2],
                  [1, 0, 0, 2, 2],
                  [1, 1, 0, 1, 2],])
d = d.to_sparse_coo()
d = transform_coo_to_csr(d)
print(d)
print("========")
print(d[[0,1,2]][:, [0,1,2]])
print("========")
# print(d[divide_index[0], divide_index[0]])
adj_sp_csr = [d[divide_index[i]][:, divide_index[i]] for i in range(len(divide_index))]
print(adj_sp_csr[0])

print("\\\\\\")
print(adj_sp_csr[1].shape)
print(torch.ones(adj_sp_csr[1].shape))
minus_adj_sp_csr = [sp.csr_matrix(torch.ones(item.shape))-item for item in adj_sp_csr]

# for i in minus_adj_sp_csr:
#     print(i)
#     print(transform_csr_to_coo(i))

print("==============")
adj = transform_csr_to_coo(adj_sp_csr[0])

print(adj)

a = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [8, 9, 10]])

print(adj.multiply(a).to_dense())
a = adj.multiply(a)
print(torch.sum(a.to_dense()))

# Acsr = sp.csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
# print('Acsr',Acsr)

# Acoo = Acsr.tocoo()
# print('Acoo',Acoo)

# Apt = torch.sparse.LongTensor(torch.LongTensor([Acoo.row.tolist(), Acoo.col.tolist()]),
#                               torch.LongTensor(Acoo.data.astype(np.int32)))
# print('Apt',Apt)
