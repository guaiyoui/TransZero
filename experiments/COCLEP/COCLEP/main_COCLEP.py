import argparse
import datetime
import os
#import random
import random

import metis
import networkx as nx
import torch
import torch as th
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, jaccard_score
import numpy as np
from torch_geometric.data import Data
from torch_sparse import spspmm, coalesce
from torch_geometric.utils import remove_self_loops, add_remaining_self_loops

from model_GCN import ConRC

def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False# 与上面一条代码配套使用，True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.enabled = False  # 禁用cudnn使用非确定性算法

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 在cuda 10.2及以上的版本中，需要设置以下环境变量来保证cuda的结果可复现
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    #torch.use_deterministic_algorithms(True) # 一些操作使用了原子操作，不是确定性算法，不能保证可复现，设置这个禁用原子操作，保证使用确定性算法

def f1_score_(comm_find, comm):

    lists = [x for x in comm_find if x in comm]
    if len(lists) == 0:
        #print("f1, pre, rec", 0.0, 0.0, 0.0)
        return 0.0, 0.0, 0.0
    pre = len(lists) * 1.0 / len(comm_find)
    rec = len(lists) * 1.0 / len(comm)
    f1 = 2 * pre * rec / (pre + rec)
    #print("f1, pre, rec", f1, pre, rec)
    return f1, pre, rec

def NMI_score(comm_find, comm, n_nodes):

    truthlabel = np.zeros((n_nodes), dtype=int)
    truthlabel[comm] = 1
    prelabel = np.zeros((n_nodes), dtype=int)
    prelabel[comm_find] = 1
    score = normalized_mutual_info_score(truthlabel, prelabel)
    #print("q, nmi:", score)
    return score

def ARI_score(comm_find, comm, n_nodes):

    truthlabel = np.zeros((n_nodes), dtype=int)
    truthlabel[comm] = 1
    prelabel = np.zeros((n_nodes), dtype=int)
    prelabel[comm_find] = 1
    score = adjusted_rand_score(truthlabel, prelabel)
    #print("q, ari:", score)

    return score

def JAC_score(comm_find, comm, n_nodes):
    truthlabel = np.zeros((n_nodes), dtype=int)
    truthlabel[comm] = 1
    prelabel = np.zeros((n_nodes), dtype=int)
    prelabel[comm_find] = 1
    score = jaccard_score(truthlabel, prelabel)
    #print("q, jac:", score)
    return score

def transform_to_binaray(comm, n_nodes):
    comm_binaray = [0 for i in range(n_nodes)]
    for item in comm:
        comm_binaray[item] = 1
    return comm_binaray

def evaluation(comm_find, comm):
    comm = torch.Tensor(comm).int()
    comm_find = torch.Tensor(comm_find).int()

    y_pred = comm_find.reshape(1, -1)
    y_true = comm.reshape(1, -1)
    pre = torch.sum(torch.multiply(y_pred, y_true))/(torch.sum(y_pred)+1E-9)
    rec = torch.sum(torch.multiply(y_pred, y_true))/(torch.sum(y_true)+1E-9)
    F1 = 2 * pre * rec / (pre + rec+1E-9)

    comm_find = comm_find.reshape(-1)
    comm = comm.reshape(-1)

    print("F1 score by COCLEP (implemented by Jianwei) is: {:.4f}".format(F1))
    print("NMI score by COCLEP (implemented by Jianwei) is: {:.4f}".format(normalized_mutual_info_score(comm_find, comm)))
    print("ARI score by COCLEP (implemented by Jianwei) is: {:.4f}".format(adjusted_rand_score(comm_find, comm)))
    print("JAC score by COCLEP (implemented by Jianwei) is: {:.4f}".format(jaccard_score(comm_find, comm)))

    return 

def validation__(val, cluster_membership, sg_nodes, n_nodes, nodes_feats, device, g, model, nodes_adj):
    scorelists = []
    for q, comm in val:
        nodeslists = sg_nodes[cluster_membership[q]]
        nodelistb = []
        if q in nodes_adj:
            neighbor = nodes_adj[q]
            nodelistb = [x for x in neighbor if x not in nodeslists]
        nodeslists = nodeslists + nodelistb
        mask = [False] * n_nodes
        for u in nodeslists:
            mask[u] = True
        feats = nodes_feats[mask].to(device)
        nodeslists = sorted(nodeslists)
        nodes_ = {}
        for i, u in enumerate(nodeslists):
            nodes_[u] = i
        sub = g.subgraph(nodeslists)
        src = []
        dst = []
        for id1, id2 in sub.edges:
            id1_ = nodes_[id1]
            id2_ = nodes_[id2]
            src.append(id1_)
            dst.append(id2_)
            src.append(id2_)
            dst.append(id1_)
        edge_index = torch.tensor([src, dst]).to(device)
        edge_index_aug, egde_attr = hypergraph_construction(edge_index, len(nodeslists), k=args.k)
        edge_index = add_remaining_self_loops(edge_index, num_nodes=len(nodeslists))[0]

        h = model((nodes_[q], None, edge_index, edge_index_aug, feats))

        numerator = torch.mm(h[nodes_[q]].unsqueeze(0), h.t())
        norm = torch.norm(h, dim=-1, keepdim=True)
        denominator = torch.mm(norm[nodes_[q]].unsqueeze(0), norm.t())
        sim = numerator / denominator
        simlists = torch.sigmoid(sim.squeeze(0)).to(
            torch.device('cpu')).numpy().tolist()  # torch.sigmoid(simlists).numpy().tolist()
        comm_ = [nodes_[x] for x in comm if x in nodeslists]
        if len(comm_)!=len(comm):
            size = len(comm)-len(comm_)
            for i in range(size):
                comm_.append(i+len(nodeslists))
        scorelists.append([nodes_[q], comm_, simlists])
    s_ = 0.1
    f1_m = 0.0
    s_m = s_
    while(s_<=0.9):
        f1_x = 0.0
        print("------------------------------", s_)
        for q, comm, simlists in scorelists:
            comm_find = []
            for i, score in enumerate(simlists):
                if score >=s_ and i not in comm_find:
                    comm_find.append(i)

            comm_find = set(comm_find)
            comm_find = list(comm_find)
            comm = set(comm)
            comm = list(comm)
            f1, pre, rec = f1_score_(comm_find, comm)
            f1_x= f1_x+f1#pre
        f1_x = f1_x/len(val)
        if f1_m<f1_x:
            f1_m = f1_x
            s_m = s_
        s_ = s_+0.05
    print("------------------------", s_m, f1_m)
    return s_m, f1_m

def loadQuerys(dataset, root, train_n, val_n, test_n, train_path, test_path, val_path):
    path_train = root + dataset + '/' + dataset + train_path
    if not os.path.isfile(path_train):
        raise Exception("No such file: %s" % path_train)
    train_lists = []
    for line in open(path_train, encoding='utf-8'):
        q, pos, comm = line.split(",")
        q = int(q)
        pos = pos.split(" ")
        pos_ = [int(x) for x in pos if int(x)!=q]
        comm = comm.split(" ")
        comm_ = [int(x) for x in comm]
        if len(train_lists)>=train_n:
            break
        train_lists.append((q, pos_, comm_))
    path_test = root + dataset + '/' + dataset + test_path
    if not os.path.isfile(path_test):
        raise Exception("No such file: %s" % path_test)
    test_lists = []
    for line in open(path_test, encoding='utf-8'):
        q, comm = line.split(",")
        q = int(q)
        comm = comm.split(" ")
        comm_ = [int(x) for x in comm]
        if len(test_lists)>=test_n:
            break
        test_lists.append((q, comm_))
    '''
    val_lists_ = test_lists[test_n:]
    test_lists = test_lists[:test_n]
    val_lists = []
    for q, comm in val_lists_:
        val_lists.append((q, comm))

    '''
    path_val = root + dataset + '/' + dataset + val_path
    if not os.path.isfile(path_val):
        raise Exception("No such file: %s" % path_val)
    val_lists = []
    for line in open(path_val, encoding='utf-8'):
        q, pos, comm = line.split(",")
        q = int(q)
        pos = pos.split(" ")
        pos_ = [int(x) for x in pos if int(x)!=q]
        comm = comm.split(" ")
        comm_ = [int(x) for x in comm]
        if len(val_lists)>=val_n:
            break
        val_lists.append((q, comm_))
    #'''

    return train_lists, val_lists, test_lists

def metis_clustering(graph, cluster_number):
    (st, parts) = metis.part_graph(graph, cluster_number)
    clusters = list(set(parts))
    cluster_membership = {}
    for i, node in enumerate(graph.nodes):
        cluster = parts[i]
        cluster_membership[node] = cluster
    return clusters, cluster_membership

def load_data(dataset, root, train_n, val_n, test_n, feats_path, cluster_number, train_path,
              test_path, val_path):
    path = root + dataset + '/' + dataset + '.txt'
    max = 0
    edges = []

    for line in open(path, encoding='utf-8'):
        node1, node2 = line.split(" ")
        node1_ = int(node1)
        node2_ = int(node2)
        if node1_==node2_:
            continue
        if max < node1_:
            max = node1_
        if max < node2_:
            max = node2_
        edges.append([node1_, node2_])
    n_nodes = max + 1
    nodeslists = [x for x in range(n_nodes)]
    graphx = nx.Graph()
    graphx.add_nodes_from(nodeslists)
    graphx.add_edges_from(edges)
    print(graphx)
    del edges

    print("---------------------cluster-------------------------------")
    clusters, cluster_membership = metis_clustering(graphx, cluster_number)
    print("---------------------cluster-------------------------------")

    nodes_adj = {}
    for id1, id2 in graphx.edges:
        if id1 not in nodes_adj:
            nodes_adj[id1] = [id2]
        else:
            nodes_adj[id1].append(id2)
        if id2 not in nodes_adj:
            nodes_adj[id2] = [id1]
        else:
            nodes_adj[id2].append(id1)

    print("---------------------graph-------------------------------")
    sg_nodes = {}

    for u, c in cluster_membership.items():
        if c not in sg_nodes:
            sg_nodes[c] = []
        sg_nodes[c].append(u)

    train, val, test = loadQuerys(dataset, root, train_n, val_n, test_n, train_path, test_path, val_path)

    print("======================featd==================================")
    path_feat = root + dataset + '/' + feats_path
    if not os.path.isfile(path_feat):
        raise Exception("No such file: %s" % path_feat)
    feats_node = {}
    count = 1
    for line in open(path_feat, encoding='utf-8'):
        if count == 1:
            node_n_, node_in_dim = line.split()
            node_in_dim = int(node_in_dim)
            count = count + 1
        else:
            emb = [float(x) for x in line.split()]
            id = int(emb[0])
            emb = emb[1:]
            feats_node[id] = emb
    nodes_feats = []

    for i in range(0, n_nodes):
        if i not in feats_node:
            nodes_feats.append([0.0] * node_in_dim)
        else:
            nodes_feats.append(feats_node[i])
    nodes_feats = th.tensor(nodes_feats)
    '''  
    nodes_feats = nodes_feats.transpose(0, 1)
    rowsum = nodes_feats.sum(1)
    rowsum[rowsum == 0] = 1
    print(rowsum)
    nodes_feats = nodes_feats / rowsum[:, np.newaxis]
    nodes_feats = nodes_feats.transpose(0, 1)
    #'''
    return nodes_feats, train, val, test, node_in_dim, n_nodes, graphx, \
           sg_nodes, clusters, cluster_membership, nodes_adj

class TwoHopNeighbor(object):
    def __call__(self, data):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        N = data.num_nodes

        value = edge_index.new_ones((edge_index.size(1), ), dtype=torch.float)

        index, value = spspmm(edge_index, value, edge_index, value, N, N, N, True)
        value.fill_(0)
        index, value = remove_self_loops(index, value)

        edge_index = torch.cat([edge_index, index], dim=1)
        if edge_attr is None:
            data.edge_index, _ = coalesce(edge_index, None, N, N)
        else:
            value = value.view(-1, *[1 for _ in range(edge_attr.dim() - 1)])
            value = value.expand(-1, *list(edge_attr.size())[1:])
            edge_attr = torch.cat([edge_attr, value], dim=0)
            data.edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)
            data.edge_attr = edge_attr

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

def hypergraph_construction(edge_index, num_nodes, k=1):
    if k == 1:
        edge_index, edge_attr = add_remaining_self_loops(edge_index, num_nodes=num_nodes)
    else:
        neighbor_augment = TwoHopNeighbor()
        hop_data = Data(edge_index=edge_index, edge_attr=None)
        hop_data.num_nodes = num_nodes
        for _ in range(k - 1):
            hop_data = neighbor_augment(hop_data)
        hop_edge_index = hop_data.edge_index
        hop_edge_attr = hop_data.edge_attr
        edge_index, edge_attr = add_remaining_self_loops(hop_edge_index, hop_edge_attr, num_nodes=num_nodes)
    return edge_index, edge_attr

def decompose_train(g, train, val, nodes_feats, n_nodes, cluster_membership,
                    sg_nodes, k, nodes_adj):
    train_lists = []
    for q, pos, comm in train:
        nodeslists = sg_nodes[cluster_membership[q]]

        nodelistb = []
        if q in nodes_adj:
            neighbor = nodes_adj[q]
            nodelistb = [x for x in neighbor if x not in nodeslists]
        nodelistb = set(nodelistb)
        nodelistb = list(nodelistb)
        nodeslists = nodeslists+nodelistb
        lists = [x for x in comm if x not in nodeslists]
        print(len(nodeslists), len(nodelistb), len(lists), len(comm))
        mask = [False]*n_nodes
        for u in nodeslists:
            mask[u] = True
        feats = nodes_feats[mask]
        nodeslists = sorted(nodeslists)
        nodes_ = {}
        for i, u in enumerate(nodeslists):
            nodes_[u]=i
        sub = g.subgraph(nodeslists)
        src = []
        dst = []
        for id1, id2 in sub.edges:
            id1_ = nodes_[id1]
            id2_ = nodes_[id2]
            src.append(id1_)
            dst.append(id2_)
            src.append(id2_)
            dst.append(id1_)
        edge_index = torch.tensor([src, dst])
        edge_index_aug, egde_attr = hypergraph_construction(edge_index, len(nodeslists), k = k)
        edge_index = add_remaining_self_loops(edge_index, num_nodes=len(nodeslists))[0]
        pos_ = [nodes_[x] for x in pos if x in nodeslists]
        #pos_ = [nodes_[x] for x in comm if x in nodeslists and x!=q]
        train_lists.append((nodes_[q], pos_, edge_index, edge_index_aug, feats))

    return train_lists

def CommunitySearch(args):


    pretime = datetime.datetime.now()
    nodes_feats, train_, val, test, node_in_dim, n_nodes, g, \
    sg_nodes, clusters, cluster_membership, nodes_adj = load_data(args.dataset, args.root,
              args.train_size, args.val_size, args.test_size,args.feats_path, args.cluster_number,
              args.train_path, args.test_path, args.val_path)

    trainlists  = decompose_train(g, train_, val, nodes_feats, n_nodes,cluster_membership,
                                            sg_nodes, args.k, nodes_adj)
    print(len(trainlists))

    print(torch.cuda.device_count())
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    #device = torch.device('cpu')
    print(device)

    train = []
    for q, pos, edge_index, edge_index_aug, feats in trainlists:
        edge_index = edge_index.to(device)
        edge_index_aug = edge_index_aug.to(device)
        feats = feats.to(device)
        train.append((q, pos, edge_index, edge_index_aug, feats))

    model = ConRC(node_in_dim, args.hidden_dim, args.num_layers, args.drop_out, args.tau,
                  device, args.alpha, args.lam, args.k)
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.to(device)
    model.reset_parameters()
    pre_process_time = (datetime.datetime.now() - pretime).seconds

    model_path = './model/' + args.dataset + '_' + args.model_path + '.pkl'

    now = datetime.datetime.now()
    optimizer.zero_grad()
    
    for epoch in range(args.epoch_n):
        model.train()
        start = datetime.datetime.now()
        loss_b = 0.0
        i = 0
        for q, pos, edge_index, edge_index_aug, feats in train:
            if len(pos) == 0:
                i = i + 1
                continue
            loss = model((q, pos, edge_index, edge_index_aug, feats))
            loss_b = loss_b + loss
            loss.backward()
            if (i + 1) % args.batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()
            i = i + 1

        epoch_time = (datetime.datetime.now() - start).seconds
        print("loss", loss_b, epoch, epoch_time)

    torch.save(model.state_dict(), model_path)  # '''
    print("------------------------evalue-------------------------")
    #if args.early_stop==1:

    model.load_state_dict(torch.load(model_path))
    model.eval()


    F1 = 0.0
    Pre = 0.0
    Rec = 0.0
    nmi_score = 0.0
    ari_score = 0.0
    jac_score = 0.0
    count = 0.0

    f1_score, precision, recall = 0.0, 0.0, 0.0
    train_model_running_time = (datetime.datetime.now() - now).seconds  # '''
    now = datetime.datetime.now()
    with th.no_grad():
        s_, f1_ = validation__(val, cluster_membership, sg_nodes,
                               n_nodes, nodes_feats, device, g, model, nodes_adj)

        print(len(test))

        comm_all, comm_find_all = [], []
        for q, comm in test:

            nodeslists = sg_nodes[cluster_membership[q]]

            nodelistb = []
            if q in nodes_adj:
                neighbor = nodes_adj[q]
                nodelistb = [x for x in neighbor if x not in nodeslists]
            nodelistb = set(nodelistb)
            nodelistb = list(nodelistb)
            nodeslists = nodeslists + nodelistb
            mask = [False] * n_nodes
            for u in nodeslists:
                mask[u] = True
            feats = nodes_feats[mask].to(device)
            nodeslists = sorted(nodeslists)
            nodes_ = {}
            for i, u in enumerate(nodeslists):
                nodes_[u] = i
            sub = g.subgraph(nodeslists)
            src = []
            dst = []
            for id1, id2 in sub.edges:
                id1_ = nodes_[id1]
                id2_ = nodes_[id2]
                src.append(id1_)
                dst.append(id2_)
                src.append(id2_)
                dst.append(id1_)
            edge_index = torch.tensor([src, dst]).to(device)
            edge_index_aug, egde_attr = hypergraph_construction(edge_index, len(nodeslists), k=args.k)
            edge_index = add_remaining_self_loops(edge_index, num_nodes=len(nodeslists))[0]

            h = model((nodes_[q], None, edge_index, edge_index_aug, feats))

            count = count + 1

            numerator = torch.mm(h[nodes_[q]].unsqueeze(0), h.t())
            norm = torch.norm(h, dim=-1, keepdim=True)
            denominator = torch.mm(norm[nodes_[q]].unsqueeze(0), norm.t())
            sim = numerator/denominator
            simlists = torch.sigmoid(sim.squeeze(0)).to(torch.device('cpu')).numpy().tolist()#torch.sigmoid(simlists).numpy().tolist()

            comm_find = []
            for i, score in enumerate(simlists):
                if score >=s_ and nodeslists[i] not in comm_find:
                    comm_find.append(nodeslists[i])
            lists = []
            for qb in nodelistb:
                if cluster_membership[qb] in lists:
                    continue
                lists.append(cluster_membership[qb])
                qb_ = nodes_[qb]
                if simlists[qb_] <s_:
                    continue
                nodeslists_ = sg_nodes[cluster_membership[qb]]

                nodelistb_ = []
                if qb in nodes_adj:
                    neighbor_ = nodes_adj[qb]
                    nodelistb_ = [x for x in neighbor_ if x not in nodeslists_]
                nodelistb_ = set(nodelistb_)
                nodelistb_ = list(nodelistb_)
                nodeslists_ = nodeslists_ + nodelistb_

                qb = q

                mask_ = [False] * n_nodes
                for u in nodeslists_:
                    mask_[u] = True
                feats = nodes_feats[mask_].to(device)
                nodeslists_ = sorted(nodeslists_)
                nodes__ = {}
                for i, u in enumerate(nodeslists_):
                    nodes__[u] = i
                sub_ = g.subgraph(nodeslists_)
                src_ = []
                dst_ = []
                for id1_, id2_ in sub_.edges:
                    id1__ = nodes__[id1_]
                    id2__ = nodes__[id2_]
                    src_.append(id1__)
                    dst_.append(id2__)
                    src_.append(id2__)
                    dst_.append(id1__)
                edge_index_ = torch.tensor([src_, dst_])
                edge_index_aug_, egde_attr_ = hypergraph_construction(edge_index_, len(nodeslists_), k=args.k)
                edge_index_ = add_remaining_self_loops(edge_index_)[0].to(device)
                h_= model((nodes__[qb], None, edge_index_, edge_index_aug_, feats))
                h_[nodes__[qb]] = h_[nodes__[qb]]+h[nodes_[q]]

                numerator_ = torch.mm(h_[nodes__[qb]].unsqueeze(0), h_.t())
                norm_ = torch.norm(h_, dim=-1, keepdim=True)
                denominator_ = torch.mm(norm_[nodes__[qb]].unsqueeze(0), norm_.t())
                sim_ = numerator_ / denominator_
                simlists_ = torch.sigmoid(sim_.squeeze(0)).to(
                    torch.device('cpu')).numpy().tolist()

                for i, score in enumerate(simlists_):
                    if score >= s_ and nodeslists_[i] not in comm_find:
                        comm_find.append(nodeslists_[i])

            comm_find = set(comm_find)
            comm_find = list(comm_find)
            comm = set(comm)
            comm = list(comm)
            comm_all = comm_all + transform_to_binaray(comm, n_nodes)
            comm_find_all = comm_find_all + transform_to_binaray(comm_find, n_nodes)
            f1, pre, rec = f1_score_(comm_find, comm)
            F1 = F1 + f1
            Pre = Pre + pre
            Rec = Rec + rec
            print("count", count)
            print("f1, pre, rec", q, f1, pre, rec)
            print("f1, pre, rec", q, F1 / count, Pre / count, Rec / count)

            nmi = NMI_score(comm_find, comm, n_nodes)
            nmi_score = nmi_score + nmi

            ari = ARI_score(comm_find, comm, n_nodes)
            ari_score = ari_score + ari

            jac = JAC_score(comm_find, comm, n_nodes)
            jac_score = jac_score + jac



    F1 = F1 / len((test))
    Pre = Pre / len((test))
    Rec = Rec / len((test))
    print("F1, Pre, Rec, s", F1, Pre, Rec, s_)


    nmi_score = nmi_score/len(test)
    print("NMI: ", nmi_score)

    ari_score = ari_score/len(test)
    print("ARI: ", ari_score)

    jac_score = jac_score/len(test)
    print("JAC: ", jac_score)

    evaluation(comm_find_all, comm_all)

    test_running_time = (datetime.datetime.now() - now).seconds
    output = args.root+'/result/'+args.dataset+args.result
    with open(output, 'a+') as fh:
        line = str(args)+" pre_process_time "+str(pre_process_time)+" train_model_running_time "+\
               str(train_model_running_time)+" test_running_time "+str(test_running_time)+" F1 "+str(F1)\
               +" Pre "+str(Pre)+" Rec "+str(Rec)+" nmi_score "+str(nmi_score)+" ari_score "+str(ari_score)\
               +" jac_score " + str(jac_score)+" F1_ "+str(f1_score)\
               +" Pre_ "+str(precision)+" Rec_ "+str(recall)
        fh.write(line + "\n")
        fh.close()
    return F1, Pre, Rec, nmi_score, ari_score, jac_score, pre_process_time, train_model_running_time, test_running_time

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='football')
    parser.add_argument('--root', type=str, default='./dataset/')
    parser.add_argument('--feats_path', type=str, default='/football_core_emb_.txt')
    parser.add_argument('--train_size', type=int, default=100)
    parser.add_argument('--val_size', type=int, default=100)
    parser.add_argument('--test_size', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--epoch_n', type=int, default=100)
    parser.add_argument('--drop_out', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)#
    parser.add_argument('--tau', type=float, default=0.2)
    parser.add_argument('--cluster_number', type=int, default=100)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--lam', type=float, default=0.2)
    parser.add_argument('--train_path', type=str, default='_3_train_pos.txt')
    parser.add_argument('--test_path', type=str, default='_test.txt')
    parser.add_argument('--val_path', type=str, default='_3_val_pos.txt')
    parser.add_argument('--model_path', type=str, default='Cluster-CLCS-')
    parser.add_argument('--count', type=int, default=1)
    parser.add_argument('--pos_size', type=int, default=3)
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--test_every', type=int, default=200)
    parser.add_argument('--result', type=str, default='_Cluster_CLCS_result.txt')


    args = parser.parse_args()

    pre_process_time_A, train_model_running_time_A, test_running_time_A = 0.0, 0.0, 0.0

    count = 0
    F1lists = []
    Prelists = []
    Reclists = []
    nmi_scorelists = []
    ari_scorelists = []
    jac_scorelists = []

    
    for i in range(args.count):
        seed_all(0)
        args.model_path='Cluster-CLCS-6'+"-"+str(count)
        count = count+1
        print('= ' * 20)
        print(count)
        now = datetime.datetime.now()
        print('##  Starting Time:', now.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
        F1, Pre, Rec, nmi_score, ari_score, jac_score, pre_process_time, train_model_running_time, test_running_time = \
            CommunitySearch(args)
        F1lists.append(F1)
        Prelists.append(Pre)
        Reclists.append(Rec)
        nmi_scorelists.append(nmi_score)
        ari_scorelists.append(ari_score)
        jac_scorelists.append(jac_score)
        pre_process_time_A = pre_process_time_A + pre_process_time
        train_model_running_time_A = train_model_running_time_A + train_model_running_time
        test_running_time_A = test_running_time_A + test_running_time
        print('## Finishing Time:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
        running_time = (datetime.datetime.now() - now).seconds
        print('## Running Time:', running_time)
        print('= ' * 20)


    F1_std = np.std(F1lists)
    F1_mean = np.mean(F1lists)
    Pre_std = np.std(Prelists)
    Pre_mean = np.mean(Prelists)
    Rec_std = np.std(Reclists)
    Rec_mean = np.mean(Reclists)
    nmi_std = np.std(nmi_scorelists)
    nmi_mean = np.mean(nmi_scorelists)
    ari_std = np.std(ari_scorelists)
    ari_mean = np.mean(ari_scorelists)
    jac_std = np.std(jac_scorelists)
    jac_mean = np.mean(jac_scorelists)

    pre_process_time_A = pre_process_time_A/float(args.count)
    train_model_running_time_A = train_model_running_time_A/float(args.count)
    test_running_time_A = test_running_time_A/float(args.count)
    output = args.root+'/result/'+args.dataset+args.result
    with open(output, 'a+') as fh:
        line = "average "+str(args)+" pre_process_time "+str(pre_process_time_A)+" train_model_running_time "+\
               str(train_model_running_time_A)+" test_running_time "+str(test_running_time_A)+\
               " F1 mean "+str(F1_mean)+" F1 std "+ str(F1_std)+" Pre mean "+str(Pre_mean)+" Pre std "+\
               str(Pre_std)+" Rec mean "+str(Rec_mean)+"Rec std "+str(Rec_std)+" nmi_score mean "+\
               str(nmi_mean)+" nmi std "+str(nmi_std)+" ari_score mean "+str(ari_mean)+" ari std "+\
               str(ari_std)+" jac mean "+str(jac_mean)+" jac std "+str(jac_std)
        fh.write(line + "\n")
        fh.close()














