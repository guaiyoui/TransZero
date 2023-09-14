from data_loader import get_dataset

import time
import utils
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from early_stop import EarlyStopping, Stop_args
from model import PretrainModel, GCN_Pretrain
from lr import PolynomialDecayLR
import os.path
import torch.utils.data as Data
from utils import *



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
    model = GCN_Pretrain(input_dim=processed_features.shape[2], config=args).to(args.device)

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

    edge_index = transform_coo_to_edge_index(adj).to(args.device)
    adj = transform_coo_to_csr(adj) # transform to csr to support slicing operation
    adj_batch, minus_adj_batch = transform_sp_csr_to_coo(adj, args.batch_size, features.shape[0]) # transform to coo to support tensor operation

    # model train
    model.train()

    print("Start train")
    t_start = time.time()

    loss_train_b = []
    for epoch in range(args.epochs):
        for index, item in enumerate(data_loader):

            start_index = index*args.batch_size
            nodes_features = (item[:, 0, :].squeeze()).to(args.device)

            adj_ = adj_batch[index].to(args.device)
            minus_adj = minus_adj_batch[index].to(args.device)
            
            # print(nodes_features.shape)
            optimizer.zero_grad()
            node_tensor = model(nodes_features, edge_index)

            # loss_train = model.contrastive_link_loss(node_tensor, edge_index)
            loss_train = model.contrastive_link_loss(node_tensor, adj_, minus_adj)
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
    model.eval()
    node_embedding = []
    for _, item in enumerate(data_loader):
        nodes_features = (item[:, 0, :].squeeze()).to(args.device)
        node_tensor = model(nodes_features, edge_index)
        if len(node_embedding) == 0:
            node_embedding = node_tensor.cpu().detach().numpy()
        else:
            new_node_embedding = node_tensor.cpu().detach().numpy()
            node_embedding = np.concatenate((node_embedding, new_node_embedding), axis=0)
    
    np.save(args.embedding_path + args.model_name + '.npy', node_embedding)