# import torch
# a = torch.Tensor([1, 2])
# b = torch.Tensor([3, 2])
# print(torch.multiply(a, 1-b))

# b[torch.argmax(a)] = 1

# c = torch.Tensor([[1, 1], [3, 4]]) 

# c[1] = b
# print(c)

# import numpy as np

# # 从.npz文件中加载数据
# data = np.load('../NAGphormer/dataset/reddit_adj.npz')

# print(data.files)


import torch

# 创建COO格式的稀疏张量
indices = torch.tensor([[0, 1, 0, 1],
                        [1, 0, 0, 1]])
values = torch.tensor([3, 4, 5, 6])
shape = torch.Size([3, 3])

sparse_tensor = torch.sparse_coo_tensor(indices, values, shape)

# 将稀疏张量转换为密集张量
dense_tensor = sparse_tensor.to_dense()

# 打印密集张量
print(dense_tensor)

node_num = 20
batch_size = 6

node_index = [i for i in range(node_num)]
divide_index = [node_index[i:i+batch_size] for i in range(0, len(node_index), batch_size)]
print(node_index, divide_index)