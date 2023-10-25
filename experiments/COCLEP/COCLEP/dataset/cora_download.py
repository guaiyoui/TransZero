import numpy as np
import os.path as osp
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from torch_geometric.datasets import Planetoid
cora = Planetoid(root='../../PU_CS/data', name='cora')

import numpy as np

label_cora = cora.y.numpy()
# 使用NumPy的unique函数获取唯一类别和每个类别的计数
unique_classes_cora, class_counts_cora = np.unique(label_cora, return_counts=True)

print('Cora dataset label:', unique_classes_cora,"\t count:" ,class_counts_cora)
cora_map = {i : cora.y[i].item() for i in range(2708)}

cora_inv_map = {}
for k, v in cora_map.items():
    if cora_inv_map.get(v) is None:
        cora_inv_map[v] = [k]
    else:
        cora_inv_map[v].append(k)

import random

def data_generation_3(d, size = 5000, unique_classes = 7):
    res = []
    count = 0
    while count < size:
        i = random.randint(0, unique_classes - 1)
        k1, k2, k3, k4 = random.sample(d[i], 4)
        res.append({"query": k1, "pos":[k2, k3, k4], "label": i})
        count += 1
    return res

def data_generation_test(d, size = 5000, unique_classes = 7):
    res = []
    count = 0
    while count < size:
        i = random.randint(0, unique_classes - 1)
        k1 = random.sample(d[i], 1)
        res.append({"query": k1[0],  "label": i})
        count += 1
    return res

res_cora = data_generation_3(cora_inv_map, size = 100, unique_classes = 7)
res_cora_val = data_generation_3(cora_inv_map, size = 100, unique_classes = 7)
res_cora_test = data_generation_test(cora_inv_map, size = 100, unique_classes = 7)

dir = 'cora/'
with open(dir + 'cora_3_train_pos.txt', "w") as f:
    for i in res_cora:
        f.write(str(i["query"]) + "," + str(i["pos"][0]) + " " + str(i["pos"][1]) + " " + str(i["pos"][2]) + "," )
        for i in cora_inv_map[i["label"]]:
            f.write(str(i) + " ")
        f.write("\n")

with open(dir + 'cora_3_val_pos.txt', "w") as f:
    for i in res_cora_val:
        f.write(str(i["query"]) + "," + str(i["pos"][0]) + " " + str(i["pos"][1]) + " " + str(i["pos"][2]) + "," )
        for i in cora_inv_map[i["label"]]:
            f.write(str(i) + " ")
        f.write("\n")

with open(dir + 'cora_test.txt', "w") as f:
    for i in res_cora_test:
        f.write(str(i["query"]) + "," )
        for i in cora_inv_map[i["label"]]:
            f.write(str(i) + " ")
        f.write("\n")

with open(dir + 'cora.txt', "w") as f:
    for i in cora.edge_index.t():
        f.write(str(i[0].item()) + " " + str(i[1].item()) + "\n")
        
with open(dir + 'cora_core_emb_.txt', "w") as f:
    cnt = 0
    f.write(str(cora.x.shape[0]) + " " + str(cora.x.shape[1]) + "\n")
    for i in cora.x:
        f.write(str(cnt) + " ")
        for j in i:
            f.write(str(jc.item()) + " ")
        cnt += 1
        f.write("\n")

with open(dir + 'cora_truth.txt', "w") as f:
    for i in sorted(cora_inv_map.keys()):
        f.write("comm" + str(i) + " ")
        for j in cora_inv_map[i]:
            f.write(str(j) + " ")
        f.write("\n")

