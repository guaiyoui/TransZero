from data_loader import get_dataset

import time
import utils
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from early_stop import EarlyStopping, Stop_args
from model import PretrainModel
from lr import PolynomialDecayLR
import os.path
import torch.utils.data as Data
from utils import *
from kmeans_pytorch import kmeans

def obtain_allnode_embedding(model, data_loader):
    model.eval()
    node_embedding = []
    for _, item in enumerate(data_loader):
        nodes_features = item.to(args.device)
        node_tensor, neighbor_tensor = model(nodes_features)
        if len(node_embedding) == 0:
            node_embedding = np.concatenate((node_tensor.cpu().detach().numpy(), neighbor_tensor.cpu().detach().numpy()), axis=1)
            # node_embedding = node_tensor.cpu().detach().numpy()
        else:
            new_node_embedding = np.concatenate((node_tensor.cpu().detach().numpy(), neighbor_tensor.cpu().detach().numpy()), axis=1)
            # new_node_embedding = node_tensor.cpu().detach().numpy()
            node_embedding = np.concatenate((node_embedding, new_node_embedding), axis=0)
    return node_embedding


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
        
        # print(node_tensor.shape, neighbor_tensor.shape)

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
        # link_loss = torch.abs(torch.sum(link_loss))
        link_loss = torch.sum(link_loss)

        TotalLoss += 0.001*link_loss
        # print(link_loss)

        return TotalLoss
    
    def contrastive_kmeanslink_loss(self, node_tensor, neighbor_tensor, adj_, minus_adj, group_connect_matrix, minus_group_connect_matrix):
        
        # print(node_tensor.shape, neighbor_tensor.shape)

        shuf_index = torch.randperm(node_tensor.shape[0])

        node_tensor_shuf = node_tensor[shuf_index] 
        neighbor_tensor_shuf = neighbor_tensor[shuf_index]

        logits_aa = torch.sigmoid(torch.sum(node_tensor * neighbor_tensor, dim = -1))
        logits_bb = torch.sigmoid(torch.sum(node_tensor_shuf * neighbor_tensor_shuf, dim = -1))
        logits_ab = torch.sigmoid(torch.sum(node_tensor * neighbor_tensor_shuf, dim = -1))
        logits_ba = torch.sigmoid(torch.sum(node_tensor_shuf * neighbor_tensor, dim = -1))
        
        TotalLoss = 0.0
        ones = torch.ones(logits_aa.size(0)).cuda(logits_aa.device)
        # TotalLoss += self.marginloss(logits_aa, logits_ba, ones)
        # TotalLoss += self.marginloss(logits_bb, logits_ab, ones)
        
        pairwise_similary = torch.mm(node_tensor, node_tensor.t())
        link_loss = minus_adj.multiply(pairwise_similary)-adj_.multiply(pairwise_similary)
        # link_loss = torch.abs(torch.sum(link_loss))
        link_loss = torch.sum(link_loss)

        # TotalLoss += 0.0000001*link_loss
        # print(link_loss)

        # group_loss = minus_group_connect_matrix.multiply(pairwise_similary)-group_connect_matrix.multiply(pairwise_similary)
        group_loss = group_connect_matrix.multiply(pairwise_similary)
        group_loss = torch.sum(group_loss)
        TotalLoss += 0.000001*group_loss

        return TotalLoss


if __name__ == "__main__":

    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    adj, features, labels, idx_train, idx_val, idx_test = get_dataset(args.dataset, args.pe_dim)

    processed_features = utils.re_features(adj, features, args.hops)  # return (N, hops+1, d)
    
    data_loader = Data.DataLoader(processed_features, batch_size=args.batch_size, shuffle = False)

    # model configuration
    model = PretrainModel(input_dim=processed_features.shape[2], config=args).to(args.device)

    print(model)
    print('total params:', sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialDecayLR(
                    optimizer,
                    warmup_updates=args.warmup_updates,
                    tot_updates=args.tot_updates,
                    lr=args.peak_lr,
                    end_lr=args.end_lr,
                    power=1.0)
    
    stopping_args = Stop_args(patience=args.patience, max_epochs=args.epochs)
    early_stopping = EarlyStopping(model, **stopping_args)

    adj = transform_coo_to_csr(adj) # transform to csr to support slicing operation
    adj_batch, minus_adj_batch = transform_sp_csr_to_coo(adj, args.batch_size, features.shape[0]) # transform to coo to support tensor operation

    # model train
    model.train()

    t_start = time.time()

    loss_train_b = []
    pseudo_assignment = []
    cluster_ids_x = []

    for epoch in range(args.epochs):
        
        if epoch % args.group_epoch_gap == 0:
            node_embedding = obtain_allnode_embedding(model, data_loader)
            if epoch == 0:
                pseudo_assignment = torch.ones(node_embedding.shape[0], 1)
            else:
                # num_communities = 2**int(epoch/args.group_epoch_gap)
                num_communities = 7
                if num_communities >= 10:
                    num_communities = 10
                cluster_ids_x, _ = kmeans(X=torch.Tensor(node_embedding), num_clusters=num_communities, distance='cosine', device=args.device)
                pseudo_assignment = construct_pseudo_assignment(cluster_ids_x)
            print(pseudo_assignment)
        for index, item in enumerate(data_loader):
            start_index = index*args.batch_size
            nodes_features = item.to(args.device)
            adj_ = adj_batch[index].to(args.device)
            minus_adj = minus_adj_batch[index].to(args.device)

            pseudo_assignment_bath = pseudo_assignment[start_index:start_index+args.batch_size].to(args.device)
            group_connect_matrix = torch.mm(pseudo_assignment_bath, pseudo_assignment_bath.t())
            minus_group_connect_matrix = 1-group_connect_matrix

            # print(nodes_features.shape)
            optimizer.zero_grad()
            node_tensor, neighbor_tensor = model(nodes_features)
            if epoch < args.group_epoch_gap or epoch > 2*args.group_epoch_gap:
            # if epoch < 10 or epoch > 20:
                loss_train = model.contrastive_link_loss(node_tensor, neighbor_tensor, adj_, minus_adj)
            else:
                loss_train = model.contrastive_kmeanslink_loss(node_tensor, neighbor_tensor, adj_, minus_adj, group_connect_matrix, minus_group_connect_matrix)
            loss_train.backward()
            optimizer.step()
            lr_scheduler.step()
            loss_train_b.append(loss_train.item())

        if early_stopping.simple_check(loss_train_b):
            break

        print('Epoch: {:04d}'.format(epoch+1),
        'loss_train: {:.4f}'.format(loss_train.item()))
        # 'loss_train: {:.4f}'.format(np.mean(np.array(loss_train_b)))
    
    print("Optimization Finished!")
    print("Train time: {:.4f}s".format(time.time() - t_start))

    # model save
    print("Start Save Model...")

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    if not os.path.exists(args.embedding_path):
        os.makedirs(args.embedding_path)

    torch.save(model.state_dict(), args.save_path + args.model_name + '.pth')
    
    # obtain all the node embedding from the learned model
    node_embedding = obtain_allnode_embedding(model, data_loader)
    
    np.save(args.embedding_path + args.model_name + '.npy', node_embedding)

    


