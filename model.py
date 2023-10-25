import torch
import math
import torch.nn as nn
from layer import TransformerBlock
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GCNConv
from utils import is_edge_in_edge_index

class PretrainModel(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.input_dim = input_dim
        self.config = config

        self.Linear1 = nn.Linear(input_dim, self.config.hidden_dim)
        self.encoder = TransformerBlock(hops=config.hops, 
                        input_dim=input_dim, 
                        n_layers=config.n_layers,
                        num_heads=config.n_heads,
                        hidden_dim=config.hidden_dim,
                        dropout_rate=config.dropout,
                        attention_dropout_rate=config.attention_dropout)
        if config.readout == "sum":
            self.readout = global_add_pool
        elif config.readout == "mean":
            self.readout = global_mean_pool
        elif config.readout == "max":
            self.readout = global_max_pool
        else:
            raise ValueError("Invalid pooling type.")
        
        self.marginloss = nn.MarginRankingLoss(0.5)

    def forward(self, x):
        node_tensor, neighbor_tensor = self.encoder(x) # (batch_size, 1, hidden_dim), (batch_size, hops, hidden_dim)
        neighbor_tensor = self.readout(neighbor_tensor, torch.tensor([0]).to(self.config.device)) # (batch_size, 1, hidden_dim)
        return node_tensor.squeeze(), neighbor_tensor.squeeze()

    def contrastive_link_loss(self, node_tensor, neighbor_tensor, adj_, minus_adj):
        

        shuf_index = torch.randperm(node_tensor.shape[0])

        node_tensor_shuf = node_tensor[shuf_index] 
        neighbor_tensor_shuf = neighbor_tensor[shuf_index]

        logits_aa = torch.sigmoid(torch.sum(node_tensor * neighbor_tensor, dim = -1))
        logits_bb = torch.sigmoid(torch.sum(node_tensor_shuf * neighbor_tensor_shuf, dim = -1))
        logits_ab = torch.sigmoid(torch.sum(node_tensor * neighbor_tensor_shuf, dim = -1))
        logits_ba = torch.sigmoid(torch.sum(node_tensor_shuf * neighbor_tensor, dim = -1))
        
        TotalLoss = 0.0
        ones = torch.ones(logits_aa.size(0)).cuda(logits_aa.device)
        TotalLoss += self.marginloss(logits_aa, logits_ba, ones)
        TotalLoss += self.marginloss(logits_bb, logits_ab, ones)
        
        pairwise_similary = torch.mm(node_tensor, node_tensor.t())
        link_loss = minus_adj.multiply(pairwise_similary)-adj_.multiply(pairwise_similary)
        # link_loss = torch.abs(torch.sum(link_loss))/(adj_.shape[0])
        link_loss = torch.sum(link_loss)/(adj_.shape[0]*adj_.shape[0])

        # TotalLoss += 0.001*link_loss
        TotalLoss += self.config.alpha*link_loss

        return TotalLoss
