REAL_DATA = {"facebook": "FacebookPagePage", "cora": "Planetoid", "citeseer": "Planetoid", "pubmed": "Planetoid",
                "chameleon": "WikipediaNetwork", "squirrel": "WikipediaNetwork", 
                "ppi": "PPI", "actor": "Actor", 
                "texas": "WebKB", "cornell": "WebKB", "wisconsin": "WebKB"}
PLANETOIDS = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}
WEBKB = {"texas": "Texas", "cornell": "Cornell", "wisconsin": "Wisconsin"}


from torch_geometric.datasets import Planetoid, WikipediaNetwork, WebKB
import torch

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

dataset_str = 'texas'

dataset = WebKB(root='./data/texas', name = "Texas")
graph = dataset[0]

# print(graph.x, graph.edge_index, graph.y)
print(graph.edge_index)
node_embedding = graph.x
# print(node_embedding[0][0:100])
node_embedding = node_embedding.type(torch.LongTensor)
# print(node_embedding[0][0:100])

torch.save([edge_index_to_sparse_coo(graph.edge_index).type(torch.LongTensor), node_embedding, graph.y.type(torch.LongTensor)], "./dataset/"+dataset_str+"_pyg.pt")
