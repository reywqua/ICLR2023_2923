import json
import math
import os
# import urllib.request
# from types import SimpleNamespace
# from urllib.error import HTTPError
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchmetrics import AUROC,Accuracy

# # pytorch lightning
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected,add_self_loops,remove_self_loops
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from func_util import *
from torchmetrics import MeanSquaredError
# PyTorch geometric
import torch_geometric
import torch_geometric.data as geo_data
import torch_geometric.nn as geo_nn
from torch_geometric.nn import GCNConv,GATConv,GINConv,GATv2Conv
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean,scatter_sum
import random
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data
from torch_geometric.transforms import Compose
import torch_geometric.transforms as T

import argparse

class EdgeIndex_Processor():
    def __init__(self, edge_index):
        super().__init__()
        self.random_walk = None
        adj,N = self.to_sparse_tensor(edge_index)
        adj_with_selfloop = self.to_sparse_tensor_with_selfloop(edge_index)
        self.N = N
        self.adj = adj.float()
        self.adj_with_loop = adj_with_selfloop.float()
        self.k_hop_neibrs = [adj.float()]
        self.calc_random_walk_matrix()

    def to_sparse_tensor(self, edge_index):
        edge_index = remove_self_loops(edge_index)[0]
        r = len(edge_index[0])
        N = edge_index.max() + 1
        t = torch.sparse_coo_tensor(edge_index, [1] * r, (N, N))
        return t, N

    def to_sparse_tensor_with_selfloop(self, edge_index):
        edge_index = add_self_loops(edge_index)[0]
        r = len(edge_index[0])
        N = edge_index.max() + 1
        t = torch.sparse_coo_tensor(edge_index, [1] * r, (N, N))
        return t

    def calc_random_walk_matrix(self):
        t = self.adj_with_loop.to_dense().sum(dim=1)
        t = 1./t
        n = len(t)
        ind = torch.tensor([[i,i] for i in range(n)]).T
        diag = torch.sparse_coo_tensor(ind,t,(n,n))
        random_walk = torch.sparse.mm(diag,self.adj)
        self.random_walk = random_walk

    def calc_random_walk_feature(self,order=10):
        t = self.random_walk
        tot_walk_feats = []
        walk_feats = []
        for i in range(self.N):
            walk_feats.append(t[i,i])
        tot_walk_feats.append(walk_feats)
        for i in range(order):
            walk_feats = []
            t = torch.sparse.mm(t,self.random_walk)
            for i in range(self.N):
                walk_feats.append(t[i, i])
            tot_walk_feats.append(walk_feats)
        tot_walk_feats = torch.tensor(tot_walk_feats).T
        return tot_walk_feats


    def calc_adj_power(self,adj, power):
        t = adj
        for _ in range(power - 1):
            t = torch.sparse.mm(t, adj)
        # set value to one
        indices = t.coalesce().indices()
        v = t.coalesce().values()
        v = torch.tensor([1 if i > 1 else i for i in v])
        diag_mask = indices[0] != indices[1]
        indices = indices[:, diag_mask]
        v = v[diag_mask]
        t = torch.sparse_coo_tensor(indices, v, (self.N, self.N))
        return t

    def postprocess_k_hop_neibrs(self,sparse_adj):
        diag = torch.diag(1. / sparse_adj.to_dense().sum(dim=1))
        diag = diag.to_sparse()
        out = torch.sparse.mm(diag, sparse_adj)
        return out


    def calc_k_hop_neibrs(self,k_hop=2):
        adj_hop_k = self.calc_adj_power(self.adj, k_hop)
        one_hop = self.k_hop_neibrs[0]
        prev_hop = self.k_hop_neibrs[1:k_hop]
        for p in prev_hop:
            one_hop += p
        final_res = adj_hop_k - one_hop

        indices = final_res.coalesce().indices()
        v = final_res.coalesce().values()
        v = [0 if i <= 0 else 1 for i in v]
        masking = []
        v_len = len(v)
        for i in range(v_len):
            if v[i] > 0:
                masking.append(i)
        v = torch.tensor(v)
        masking = torch.tensor(masking).long()
        indices = indices[:, masking]
        v = v[masking]
        final_res = torch.sparse_coo_tensor(indices, v, (self.N, self.N))
        return final_res


    def run(self,k_hop=[2,3],random_walk_order=10):
        walk_feature = self.calc_random_walk_feature(order=random_walk_order)
        for k in k_hop:
            t = self.calc_k_hop_neibrs(k)
            self.k_hop_neibrs.append(t.float())
        # normed_k_hop_adj = [self.postprocess_k_hop_neibrs(i.float()) for i in self.k_hop_neibrs]   # 是否使用D^-1*A
        return self.k_hop_neibrs,walk_feature


class Synthetic_Dataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        print(self.processed_paths)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def create_new_data(self, data_list):
        labels = [i[0] for i in data_list]
        N = len(labels)
        X = torch.tensor([[1.0] * 5] * N)
        edges = []
        for idx, d in enumerate(data_list):
            for q in d[2:]:
                edges.append([idx, q])
                edges.append([q, idx])

        edge_index = torch.tensor(edges).t().contiguous()
        edge_index = torch.unique(edge_index, dim=1)
        k_hop_neibrs, random_walk_feats = EdgeIndex_Processor(edge_index).run()
        hop1, hop2, hop3 = k_hop_neibrs[0], k_hop_neibrs[1], k_hop_neibrs[2]
        hop1 = hop1.coalesce().indices().tolist()
        hop2 = hop2.coalesce().indices().tolist()
        hop3 = hop3.coalesce().indices().tolist()
        return Data(edge_index=edge_index, x=X, y=labels, rand_feature=random_walk_feats, hop1=hop1, hop2=hop2,
                    hop3=hop3)

    def collate_graph_adj(self, edge_list, ptr):
        edges = torch.cat([torch.tensor(i) + ptr[idx] for idx, i in enumerate(edge_list)], dim=1)
        N = ptr[-1]
        return torch.sparse_coo_tensor(edges, [1.] * edges.shape[1], (N, N))

    def process_data_list(self, file_path):
        with open(file_path, 'r') as f:
            data_list = []
            pyg_data_list = []
            for i in f.readlines():
                q = i.replace('\n', '').split(' ')
                q = [int(i) for i in q]
                if q[0] > 10:
                    if len(data_list) > 0:
                        pyg_data = self.create_new_data(data_list)
                        pyg_data_list.append(pyg_data)
                    data_list = []
                else:
                    data_list.append(q)
        return pyg_data_list

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        print('process')
        # Read data into huge `Data` list.
        data_list = self.process_data_list('data/TRIANGLE_EX/TRIANGLE_EX_test.txt')  # change the path here if want to experiment with LCC(X) !!!!!
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



class Model(nn.Module):
    def __init__(self,in_dim, hidden1_dim,hidden2_dim,layer_name='gcn',head_num=16,random_walk_dim=5,k_hop_adj=None,random_walk_feats=None,num_classes=6,use_base_gnn = False,use_random_walk = True):
        super().__init__()
        self.gnn_model = GNN(in_dim, hidden1_dim,hidden2_dim,layer_name=layer_name,head_num=head_num)
        if use_random_walk:
            self.subgraph_model = Subgraph_GNN(in_dim+random_walk_dim+1, hidden1_dim,hidden2_dim,k_hop_adj,random_walk_feats,use_rw= use_random_walk)
        else:
            self.subgraph_model = Subgraph_GNN(in_dim, hidden1_dim, hidden2_dim, k_hop_adj, random_walk_feats, use_rw=use_random_walk)
        # self.cls_head = nn.Linear(hidden2_dim*2,num_classes)
        self.use_rw = use_random_walk
        self.use_gnn = use_base_gnn

    def forward(self,x,edges,walk_feats,hop1,hop2,hop3):
        if self.use_gnn:
            gnn_output = self.gnn_model(x,edges)
        subgraph_model_out = self.subgraph_model(x,walk_feats,hop1,hop2,hop3)
        if self.use_gnn:
            h = torch.cat((gnn_output,subgraph_model_out),dim=1)
            return h
        else:
            return subgraph_model_out




class GNN(nn.Module):
    def __init__(self,in_dim, hidden1_dim,hidden2_dim,layer_name='gcn',head_num=16):
        super().__init__()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        if layer_name=='gcn':
            self.gcn1 = GCNConv(in_dim, hidden1_dim,add_self_loops=False)
            self.gcn2 = GCNConv(hidden1_dim, hidden2_dim,add_self_loops=False)
        elif layer_name=='gat':
            self.gcn1 = GATConv(in_dim, hidden1_dim,heads = head_num,concat=False,dropout=0.5)
            self.gcn2 = GATConv(hidden1_dim, hidden2_dim,heads = head_num,concat=False,dropout=0.5)
        elif layer_name=='gin':
            nn_callable1 = nn.Sequential(nn.Linear(in_dim,hidden1_dim),nn.ReLU(),nn.Dropout(),nn.Linear(hidden1_dim,hidden1_dim),nn.ReLU())
            nn_callable2 = nn.Sequential(nn.Linear(hidden1_dim,hidden2_dim),nn.ReLU(),nn.Dropout(),nn.Linear(hidden2_dim,hidden2_dim),nn.ReLU())
            nn_callable3 = nn.Sequential(nn.Linear(hidden2_dim,hidden2_dim),nn.ReLU(),nn.Dropout(),nn.Linear(hidden2_dim,hidden2_dim),nn.ReLU())
            nn_callable4 = nn.Sequential(nn.Linear(hidden2_dim,hidden2_dim),nn.ReLU(),nn.Dropout(),nn.Linear(hidden2_dim,hidden2_dim),nn.ReLU())
            self.gcn1 = GINConv(nn=nn_callable1)
            self.gcn2 = GINConv(nn=nn_callable2)
            self.gcn3 = GINConv(nn=nn_callable3)
            self.gcn4 = GINConv(nn=nn_callable4)
        else:
            print ('gnn module error')


    def forward(self,x,edges):
        h = self.gcn1(x,edges)
        h = self.gcn2(h,edges)
        h = self.gcn3(h,edges)
        h = self.gcn4(h,edges)
        return h


class Subgraph_GNN(nn.Module):
    def __init__(self,in_dim, hidden1_dim,hidden2_dim, k_hop_adj=[0,0,0,0],random_walk_feats=None,use_rw = True):
        super().__init__()
        self.layer0 = nn.Sequential(nn.Linear(in_dim,hidden1_dim),nn.ReLU(),nn.Linear(hidden1_dim,hidden2_dim))
        self.layer1 = nn.Sequential(nn.Linear(in_dim, hidden1_dim), nn.ReLU(), nn.Linear(hidden1_dim, hidden2_dim))
        self.layer2 = nn.Sequential(nn.Linear(in_dim, hidden1_dim), nn.ReLU(), nn.Linear(hidden1_dim, hidden2_dim))
        self.layer3 = nn.Sequential(nn.Linear(in_dim, hidden1_dim), nn.ReLU(), nn.Linear(hidden1_dim, hidden2_dim))
        # self.hop1,self.hop2,self.hop3 = k_hop_adj[0],k_hop_adj[1],k_hop_adj[2]
        # self.rand = random_walk_feats
        self.use_rw = use_rw


    def forward(self, x,walk_feats,hop1,hop2,hop3):
        if self.use_rw:
            X = torch.cat((x,walk_feats),dim=1)
        else:
            X = x
        hop0_out = self.layer0(X)
        hop1_out = self.layer1(hop1.matmul(X))
        hop2_out = self.layer2(hop2.matmul(X))
        hop3_out = self.layer3(hop3.matmul(X))
        return hop0_out+hop1_out+hop2_out+hop3_out


class PL_GCN(pl.LightningModule):
    def __init__(self,in_dim, hidden1_dim,hidden2_dim,layer_name='gcn',head_num=16,random_walk_dim=10,num_classes=6,lr=1e-2,weight_decay=2e-3,use_benchamark=False,node_classification=False,use_gnn=False,use_rw=True,task_id=0):
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        #         self.save_hyperparameters()
        # Create model
        self.task_id = task_id
        self.save_hyperparameters()
        self.lr = lr
        self.node_cls = node_classification
        self.weight_decay = weight_decay
        self.benchmark = use_benchamark
        self.mse = MeanSquaredError()
        if use_benchamark:
            self.model = GNN(in_dim,hidden1_dim,hidden2_dim,layer_name,head_num)
            self.cls = nn.Linear(hidden2_dim,num_classes)
        else:
            if use_gnn:
                self.cls = nn.Linear(2*hidden2_dim, num_classes)
            else:
                self.cls = nn.Linear(hidden2_dim, num_classes)
            self.args = {'in_dim': in_dim, 'hidden1_dim': hidden1_dim, 'hidden2_dim': hidden2_dim,
                         'layer_name': layer_name, 'random_walk_dim': random_walk_dim, 'num_classes': num_classes,'use_base_gnn':use_gnn,'use_random_walk':use_rw}
            self.model = Model(**self.args)

        self.log_prob_nn = nn.LogSoftmax(dim=-1)
        self.eval_roc = AUROC(num_classes=num_classes)


    # def set_data(self, pyg_dataset):
    #     self.pyg_fulledge_dataset = pyg_dataset
    #     self.pyg_data = pyg_dataset
    #     self.calc_second_order_adj()


    def collate_graph_adj(self,edge_list, ptr):
        # print ('######################',self.device)
        edges = torch.cat([torch.tensor(i).to(self.device) + ptr[idx] for idx, i in enumerate(edge_list)], dim=1)
        N = ptr[-1]
        val = torch.tensor([1.] * edges.shape[1]).to(self.device)
        return torch.sparse_coo_tensor(edges, val, (N, N)).to(self.device)

    def forward(self,x,edges):
        # Forward function that is run when visualizing the graph
        h = self.model(x, edges)
        return h

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=0)
        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,20,1e-3)
        return [optimizer]

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        #         print (batch_idx)
        if not self.benchmark:
            hop1=self.collate_graph_adj(batch.hop1,batch.ptr)
            hop2 = self.collate_graph_adj(batch.hop2, batch.ptr)
            hop3 = self.collate_graph_adj(batch.hop3, batch.ptr)
            h = self.model(batch.x, batch.edge_index,batch.rand_feature,hop1,hop2,hop3)
        else:
            h = self.model(batch.x, batch.edge_index)

        h = self.log_prob_nn(self.cls(h))
        y = torch.tensor(batch.y).view(-1,).to(self.device)
        loss_val = F.nll_loss(h, y)

        self.log("train_loss", loss_val.item(), prog_bar=True)

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        #         self.log("train_acc", acc,prog_bar=True, logger=True)
        #         self.log("train_loss", loss,prog_bar=True, logger=True)
        #         self.logger.experiment.add_scalar('tree_em_loss/train',loss.item(),self.global_step)
        return loss_val  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        if not self.benchmark:
            hop1=self.collate_graph_adj(batch.hop1,batch.ptr)
            hop2 = self.collate_graph_adj(batch.hop2, batch.ptr)
            hop3 = self.collate_graph_adj(batch.hop3, batch.ptr)
            h = self.model(batch.x, batch.edge_index,batch.rand_feature,hop1,hop2,hop3)
        else:
            h = self.model(batch.x, batch.edge_index)

        h = self.log_prob_nn(self.cls(h))
        y = torch.tensor(batch.y).view(-1,).to(self.device)
        loss_val = F.nll_loss(h, y)
        # By default logs it per epoch (weighted average over batches)
        self.log("val_loss", loss_val.item(), prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        if not self.benchmark:
            hop1=self.collate_graph_adj(batch.hop1,batch.ptr)
            hop2 = self.collate_graph_adj(batch.hop2, batch.ptr)
            hop3 = self.collate_graph_adj(batch.hop3, batch.ptr)
            h = self.model(batch.x, batch.edge_index,batch.rand_feature,hop1,hop2,hop3)
        else:
            h = self.model(batch.x, batch.edge_index)

        h = self.cls(h)
        y = torch.tensor(batch.y).view(-1,).to(self.device)
        # By default logs it per epoch (weighted average over batches)
        roc_auc = self.eval_roc(h,y)
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log_dict({'rocauc':roc_auc})



# class Norm_y(object):
#     def __init__(self,y_std = torch.tensor([[ 3.0723, 25.9458, 17.7789,  6.9390,  0.1112]])):
#         self.y_std = y_std
#     def __call__(self,data):
#         y = data.y
#         y = y/self.y_std
#         data.y = y
#         x = data.x
#         x[:,1] = x[:,1]/6.
#         x = x[:,1:]
#         data.x = x
#         return data


if __name__=='__main__':
    # dset = Planetoid(name='CiteSeer', root='data/citeseer/')
    # model = GNN(3703, 16, 15)
    # out = model(dset.data.x, dset.data.edge_index)
    # edges = torch.tensor([[0, 1, 0, 2, 1, 3, 2, 3], [1, 0, 2, 0, 3, 1, 3, 2]]).long()

    parser = argparse.ArgumentParser(description='lcc or tri')
    parser.add_argument('--use_gnn', type=int, default=0,
                        help='whether use benchmark method')
    parser.add_argument('--is_lcc', type=int, default=1,
                        help='choose dataset')
    args = parser.parse_args()

    results = []
    lcc_or_trianle = args.is_lcc
    if lcc_or_trianle:
        lcc_train = Synthetic_Dataset("data/pyg_LCC_EX/train")
        lcc_test = Synthetic_Dataset("data/pyg_LCC_EX/test")
        indices = [i for i in range(999)]
        random.shuffle(indices)
        num_class=3
    else:
        lcc_train = Synthetic_Dataset("data/pyg_TRIANGLE_EX/train")
        lcc_test = Synthetic_Dataset("data/pyg_TRIANGLE_EX/test")
        indices = [i for i in range(999)]
        random.shuffle(indices)
        num_class=2
        print ('&&&&&&&&&&&&&&&&&&&&&&&')
    for i in range(10):
        seed = np.random.randint(5000, 60000)
        pl.seed_everything(seed)
        train_lcc = lcc_train[indices[:800]]
        valid_lcc = lcc_train[indices[800:]]
        train_loader = DataLoader(train_lcc, batch_size=32, shuffle=True)
        val_loader = DataLoader(valid_lcc, batch_size=100, shuffle=False)
        test_loader = DataLoader(lcc_test, batch_size=100, shuffle=False)

        num_feats = 5
        hidden1 = 48
        hidden2 = 32
        use_gnn = True if args.use_gnn==1 else False
        use_benchmark = False
        pl_model = PL_GCN(num_feats, hidden1, hidden2, num_classes=num_class, use_benchamark=use_benchmark, random_walk_dim=10,node_classification=False, lr=8e-3, layer_name='gin',use_gnn=use_gnn,use_rw=True)
        trainer = pl.Trainer(default_root_dir=f'saved_models/lcc/', gpus=1 if torch.cuda.is_available() else 0, max_epochs=100,
                             callbacks=[EarlyStopping(patience=50, monitor='val_loss', mode='min'),
                                        ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss")])
        trainer.fit(model=pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        model = pl_model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        res = trainer.test(model=model, dataloaders=test_loader)[0]
        res['task']=lcc_or_trianle
        res['iteration'] = i
        results.append(res)
    print (results)


