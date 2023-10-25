
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_geometric.nn import HypergraphConv, GCNConv, APPNP
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import to_dense_adj
from torch_scatter import scatter


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.fcs = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.fcs(x)


class Contra(nn.Module):
    def __init__(self, device):
        super(Contra, self).__init__()
        self.device = device

    def forward(self, h, h_aug, tau, train, alpha, lam, edge_index):
        if self.training == False:
            return h

        q, pos = train

        numerator1 = torch.mm(h[q].unsqueeze(0), h.t())
        norm1 = torch.norm(h, dim=-1, keepdim=True)
        denominator1 = torch.mm(norm1[q].unsqueeze(0), norm1.t())
        sim1_ = numerator1 / denominator1
        sim1 = torch.exp(sim1_ / tau)

        numerator_aug1 = torch.mm(h_aug[q].unsqueeze(0), h_aug.t())
        norm_aug1 = torch.norm(h_aug, dim=-1, keepdim=True)
        denominator_aug1 = torch.mm(norm_aug1[q].unsqueeze(0), norm_aug1.t())
        sim_aug1_ = numerator_aug1 / denominator_aug1
        sim_aug1 = torch.exp(sim_aug1_ / tau)

        numerator2 = torch.mm(h[q].unsqueeze(0), h_aug.t())
        denominator2 = torch.mm(norm1[q].unsqueeze(0), norm_aug1.t())
        sim2_ = numerator2 / denominator2
        sim2 = torch.exp(sim2_ / tau)

        numerator_aug2 = torch.mm(h_aug[q].unsqueeze(0), h.t())
        denominator_aug2 = torch.mm(norm_aug1[q].unsqueeze(0), norm1.t())
        sim_aug2_ = numerator_aug2 / denominator_aug2
        sim_aug2 = torch.exp(sim_aug2_ / tau)

        mask_p = [False] * h.shape[0]
        mask_p = torch.tensor(mask_p)
        mask_p.to(self.device)
        mask_p[pos] = True
        mask_p[q] = False

        z1q = torch.tensor([0.0]).to(self.device)
        z_aug1q = torch.tensor([0.0]).to(self.device)
        z2q = torch.tensor([0.0]).to(self.device)
        z_aug2q = torch.tensor([0.0]).to(self.device)

        if len(pos) != 0:
            z1q = sim1.squeeze(0)[mask_p] / (torch.sum(sim1.squeeze(0)))
            z1q = -torch.log(z1q).mean()
            z_aug1q = sim_aug1.squeeze(0)[mask_p] / (torch.sum(sim_aug1.squeeze(0)))
            z_aug1q = -torch.log(z_aug1q).mean()

            z2q = sim2.squeeze(0)[mask_p] / (torch.sum(sim2.squeeze(0)))
            z2q = -torch.log(z2q).mean()
            z_aug2q = sim_aug2.squeeze(0)[mask_p] / (torch.sum(sim_aug2.squeeze(0)))
            z_aug2q = -torch.log(z_aug2q).mean()  # '''


        loss_intra = 0.5*(z1q+z_aug1q)
        loss_inter = 0.5*(z2q+z_aug2q)
        z_unsup = -torch.log(sim2.squeeze(0)[q]/torch.sum(sim2.squeeze(0)))
        z_aug_unsup = -torch.log(sim_aug2.squeeze(0)[q]/torch.sum(sim_aug2.squeeze(0)))
        loss_unsup = 0.5 * z_unsup + 0.5 * z_aug_unsup  # '''

        loss = (loss_intra+ alpha *loss_inter) + lam * loss_unsup #+ loss2r * lc

        # adj = to_dense_adj(edge_index, max_num_nodes=h.shape[0])[0]
        # deg = torch.sum(adj, 1).unsqueeze(1)
        # pm = torch.sigmoid(sim1_)
        # pm_ = torch.mm(pm, torch.t(torch.ones(pm.shape[0], pm.shape[1]).to(self.device) - pm))
        # loss_c = torch.sum(torch.sum(torch.mul(adj, pm_))) / (torch.sum(torch.sum(torch.mul(pm, deg))))

        return loss#+loss_c


class ConRC(nn.Module):
    def __init__(self, node_in_dim, hidden_dim, num_layers, dropout, tau, device, alpha, lam, k):

        super(ConRC, self).__init__()
        self.tau = tau
        self.alpha = alpha
        self.lam = lam
        self.dropout = dropout
        self.num_layers = num_layers
        self.device = device
        self.k = k

        self.contra = Contra(device)

        self.layersq = nn.ModuleList()
        self.layersq.append(GCNConv(1, hidden_dim))
        for _ in range(num_layers - 1):
            self.layersq.append(GCNConv(hidden_dim, hidden_dim))
        self.layershq = nn.ModuleList()
        self.layershq.append(HypergraphConv(1, hidden_dim))
        for _ in range(num_layers - 1):
            self.layershq.append(HypergraphConv(hidden_dim, hidden_dim))

        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(node_in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        self.layersh = nn.ModuleList()
        self.layersh.append(HypergraphConv(node_in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layersh.append(HypergraphConv(hidden_dim, hidden_dim))

        self.layersf = nn.ModuleList()
        self.layersf.append(GCNConv(hidden_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layersf.append(GCNConv(hidden_dim, hidden_dim))
        self.layersfh = nn.ModuleList()
        self.layersfh.append(HypergraphConv(hidden_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layersfh.append(HypergraphConv(hidden_dim, hidden_dim))


        self.mlp1 = MLP(hidden_dim, hidden_dim)#'''

        self.att_weightsq = []
        self.att_weights = []
        self.att_weighthsq = []
        self.att_weighths = []
        for _ in range(num_layers):
            att_weightq = nn.Parameter(torch.empty(size=(hidden_dim, 1)), requires_grad=True)
            glorot(att_weightq)
            att_weight = nn.Parameter(torch.empty(size=(hidden_dim, 1)), requires_grad=True)
            glorot(att_weight)
            att_weighthq = nn.Parameter(torch.empty(size=(hidden_dim, 1)), requires_grad=True)
            glorot(att_weighthq)
            att_weighth = nn.Parameter(torch.empty(size=(hidden_dim, 1)), requires_grad=True)
            glorot(att_weighth)
            self.att_weightsq.append(att_weightq)
            self.att_weighthsq.append(att_weighthq)
            self.att_weights.append(att_weight)
            self.att_weighths.append(att_weighth)

        self.att_weightq_ = nn.Parameter(torch.empty(size=(hidden_dim, 1)), requires_grad=True)
        glorot(self.att_weightq_)
        self.att_weight_ = nn.Parameter(torch.empty(size=(hidden_dim, 1)), requires_grad=True)
        glorot(self.att_weight_)
        self.att_weighthq_ = nn.Parameter(torch.empty(size=(hidden_dim, 1)), requires_grad=True)
        glorot(self.att_weighthq_)
        self.att_weighth_ = nn.Parameter(torch.empty(size=(hidden_dim, 1)), requires_grad=True)
        glorot(self.att_weighth_)

        self.linerquerys = torch.nn.Linear(1, hidden_dim)
        self.linerfeats = torch.nn.Linear(node_in_dim, hidden_dim)

    def reset_parameters(self):
        print("reset")

    def attetion_layerq(self, x, layer):
        return torch.matmul(x, self.att_weightsq[layer].to(self.device))

    def attetion_layerhq(self, x, layer):
        return torch.matmul(x, self.att_weighthsq[layer].to(self.device))

    def attetion_layerq_(self, x):
        return torch.matmul(x, self.att_weightq_.to(self.device))

    def attetion_layerhq_(self, x):
        return torch.matmul(x, self.att_weighthq_.to(self.device))

    def attetion_layer(self, x, layer):
        return torch.matmul(x, self.att_weights[layer].to(self.device))

    def attetion_layerh(self, x, layer):
        return torch.matmul(x, self.att_weighths[layer].to(self.device))

    def attetion_layer_(self, x):
        return torch.matmul(x, self.att_weight_.to(self.device))

    def attetion_layerh_(self, x):
        return torch.matmul(x, self.att_weighth_.to(self.device))

    def hyperedge_representation(self, x, edge_index):
        #h = self.mlp2(x)
        h = x#torch.tanh(self.att(x))
        edges = h[edge_index[0]]
        nodes = h[edge_index[1]]

        sim = torch.exp(torch.cosine_similarity(edges, nodes))

        denominator = scatter(sim, edge_index[1], dim=0, reduce='sum')
        denominator = denominator[edge_index[1]]
        sim = (sim/denominator).unsqueeze(1)

        edges_ = x[edge_index[0]]
        edges_ = sim * (edges_)

        hyperedge = scatter(edges_, edge_index[1], dim=0, reduce='sum')
        #hyperedge = torch.cat([x, hyperedge], 1)

        return hyperedge

    def compute_loss(self, train):
        loss = None
        q, pos, edge_index, edge_index_aug, feats = train
        querys = torch.zeros(feats.shape[0], 1).to(self.device)
        querys[q] = 1.0

        hq = F.relu(self.layersq[0](querys, edge_index)).to(self.device)
        h = F.relu(self.layers[0](feats, edge_index)).to(self.device)
        atten_co = torch.cat([self.attetion_layerq(hq, 0), self.attetion_layer(h, 0)], 1)
        atten_co = F.softmax(atten_co, dim=1).unsqueeze(2)
        hf = torch.stack([hq, h], dim=1)
        hf = atten_co * hf
        hf = torch.sum(hf, dim=1)

        h_augq = F.relu(self.layershq[0](querys, edge_index_aug))
        h_aug = F.relu(self.layersh[0](feats, edge_index_aug))
        atten_coh = torch.cat([self.attetion_layerhq(h_augq, 0), self.attetion_layerh(h_aug, 0)], 1)
        atten_coh = F.softmax(atten_coh, dim=1).unsqueeze(2)
        h_augf = torch.stack([h_augq, h_aug], dim=1)
        h_augf = atten_coh * h_augf
        h_augf = torch.sum(h_augf, dim=1)

        querys = self.linerquerys(querys)
        feats = self.linerfeats(feats)

        atten_co_ = torch.cat([self.attetion_layerq_(querys), self.attetion_layer_(feats)], 1)
        atten_co_ = F.softmax(atten_co_, dim=1).unsqueeze(2)
        hf_ = torch.stack([querys, feats], dim=1)
        hf_ = atten_co_ * hf_
        hf_ = torch.sum(hf_, dim=1)
        hf = F.relu(hf + self.layersf[0](hf_, edge_index))

        atten_coh_ = torch.cat([self.attetion_layerhq_(querys), self.attetion_layerh_(feats)], 1)
        atten_coh_ = F.softmax(atten_coh_, dim=1).unsqueeze(2)
        hfh_ = torch.stack([querys, feats], dim=1)
        hfh_ = atten_coh_ * hfh_
        hfh_ = torch.sum(hfh_, dim=1)
        h_augf = F.relu(h_augf + self.layersfh[0](hfh_, edge_index_aug))

        for _ in range(self.num_layers - 2):

            hq = F.dropout(hq, training=self.training, p=self.dropout)
            h = F.dropout(h, training=self.training, p=self.dropout)
            hf = F.dropout(hf, training=self.training, p=self.dropout)
            h_augq = F.dropout(h_augq, training=self.training, p=self.dropout)
            h_aug = F.dropout(h_aug, training=self.training, p=self.dropout)
            h_augf = F.dropout(h_augf, training=self.training, p=self.dropout)

            hq = F.relu(self.layersq[_+1](hq, edge_index))
            h = F.relu(self.layers[_+1](h, edge_index))
            atten_co = torch.cat([self.attetion_layerq(hq, _+1), self.attetion_layer(h, _+1)], 1)
            atten_co = F.softmax(atten_co, dim=1).unsqueeze(2)
            hfx = torch.stack([hq, h], dim=1)
            hfx = atten_co * hfx
            hfx = torch.sum(hfx, dim=1)
            hf = F.relu(hfx + self.layersf[_+1](hf, edge_index))

            h_augq = F.relu(self.layershq[_+1](h_augq, edge_index_aug))
            h_aug = F.relu(self.layersh[_+1](h_aug, edge_index_aug))
            atten_coh = torch.cat([self.attetion_layerhq(h_augq, _+1), self.attetion_layerh(h_aug, _+1)], 1)
            atten_coh = F.softmax(atten_coh, dim=1).unsqueeze(2)
            h_augfx = torch.stack([h_augq, h_aug], dim=1)
            h_augfx = atten_coh * h_augfx
            h_augfx = torch.sum(h_augfx, dim=1)
            h_augf = F.relu(h_augfx + self.layersfh[_+1](h_augf, edge_index_aug))

        hq = F.dropout(hq, training=self.training, p=self.dropout)
        h = F.dropout(h, training=self.training, p=self.dropout)
        hf = F.dropout(hf, training=self.training, p=self.dropout)
        h_augq = F.dropout(h_augq, training=self.training, p=self.dropout)
        h_aug = F.dropout(h_aug, training=self.training, p=self.dropout)
        h_augf = F.dropout(h_augf, training=self.training, p=self.dropout)

        hq = self.layersq[self.num_layers - 1](hq, edge_index)
        h = self.layers[self.num_layers - 1](h, edge_index)
        atten_co = torch.cat([self.attetion_layerq(hq, self.num_layers-1), self.attetion_layer(h, self.num_layers-1)], 1)
        atten_co = F.softmax(atten_co, dim=1).unsqueeze(2)
        hfx = torch.stack([hq, h], dim=1)
        hfx = atten_co * hfx
        hfx = torch.sum(hfx, dim=1)
        hf = hfx + self.layersf[self.num_layers - 1](hf, edge_index)

        h_augq = self.layershq[self.num_layers - 1](h_augq, edge_index_aug)
        h_aug = self.layersh[self.num_layers - 1](h_aug, edge_index_aug)
        atten_coh = torch.cat([self.attetion_layerhq(h_augq, self.num_layers-1), self.attetion_layerh(h_aug, self.num_layers-1)], 1)
        atten_coh = F.softmax(atten_coh, dim=1).unsqueeze(2)
        h_augfx = torch.stack([h_augq, h_aug], dim=1)
        h_augfx = atten_coh * h_augfx
        h_augfx = torch.sum(h_augfx, dim=1)
        h_augf = h_augfx + self.layersfh[self.num_layers - 1](h_augf, edge_index_aug)


        #h_ = self.mlp1(torch.cat([hf, hf], 1))
        h_ = self.mlp1(hf)
        h_auge = self.hyperedge_representation(h_augf, edge_index_aug)
        #h_auge = self.lineraugh(h_auge)#'''
        h_auge = self.mlp1(h_auge)  # '''

        if loss is None:
            loss = self.contra(h_, h_auge, self.tau, (q, pos), self.alpha, self.lam, edge_index)
        else:
            loss = loss + self.contra(h_, h_auge, self.tau, (q, pos), self.alpha, self.lam, edge_index)


        return loss

    def forward(self, train):

        if self.training==False:
            q, pos, edge_index, edge_index_aug, feats = train
            querys = torch.zeros(feats.shape[0], 1).to(self.device)
            querys[q] = 1.0

            hq = F.relu(self.layersq[0](querys, edge_index))
            h = F.relu(self.layers[0](feats, edge_index))
            atten_co = torch.cat([self.attetion_layerq(hq, 0), self.attetion_layer(h, 0)], 1)
            atten_co = F.softmax(atten_co, dim=1).unsqueeze(2)
            hf = torch.stack([hq, h], dim=1)
            hf = atten_co * hf
            hf = torch.sum(hf, dim=1)

            querys = self.linerquerys(querys)
            feats = self.linerfeats(feats)

            atten_co_ = torch.cat([self.attetion_layerq_(querys), self.attetion_layer_(feats)], 1)
            atten_co_ = F.softmax(atten_co_, dim=1).unsqueeze(2)
            hf_ = torch.stack([querys, feats], dim=1)
            hf_ = atten_co_ * hf_
            hf_ = torch.sum(hf_, dim=1)
            hf = F.relu(hf + self.layersf[0](hf_, edge_index))

            for _ in range(self.num_layers - 2):
                hq = F.dropout(hq, training=self.training, p=self.dropout)
                h = F.dropout(h, training=self.training, p=self.dropout)
                hf = F.dropout(hf, training=self.training, p=self.dropout)

                hq = F.relu(self.layersq[_ + 1](hq, edge_index))
                h = F.relu(self.layers[_ + 1](h, edge_index))
                atten_co = torch.cat([self.attetion_layerq(hq, _ + 1), self.attetion_layer(h, _ + 1)], 1)
                atten_co = F.softmax(atten_co, dim=1).unsqueeze(2)
                hfx = torch.stack([hq, h], dim=1)
                hfx = atten_co * hfx
                hfx = torch.sum(hfx, dim=1)
                hf = F.relu(hfx + self.layersf[_ + 1](hf, edge_index))

            hq = F.dropout(hq, training=self.training, p=self.dropout)
            h = F.dropout(h, training=self.training, p=self.dropout)
            hf = F.dropout(hf, training=self.training, p=self.dropout)

            hq = self.layersq[self.num_layers - 1](hq, edge_index)
            h = self.layers[self.num_layers - 1](h, edge_index)
            atten_co = torch.cat(
                [self.attetion_layerq(hq, self.num_layers - 1), self.attetion_layer(h, self.num_layers - 1)], 1)
            atten_co = F.softmax(atten_co, dim=1).unsqueeze(2)
            hfx = torch.stack([hq, h], dim=1)
            hfx = atten_co * hfx
            hfx = torch.sum(hfx, dim=1)
            hf = hfx + self.layersf[self.num_layers - 1](hf, edge_index)

            h_ = self.mlp1(hf)
            return h_

        loss = self.compute_loss(train)

        return loss

