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
import pickle
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
from tudataset import TU_Dataset

# PyTorch geometric
import torch_geometric
import torch_geometric.data as geo_data
import torch_geometric.nn as geo_nn
from torch_geometric.nn import GCNConv,GATConv,GINConv,SAGEConv
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean,scatter_sum

import argparse

use_cuda = True if torch.cuda.is_available() else False
gpu_num = 1 if torch.cuda.is_available() else 0

class Model(nn.Module):
    def __init__(self,in_dim, hidden1_dim,hidden2_dim,layer_name='gcn',head_num=16,random_walk_dim=5,k_hop_adj=None,random_walk_feats=None,num_classes=6,dropout=0.,layer_num=3,subgraph_layer_num=3):
        super().__init__()
        self.gnn_model = GNN(in_dim, hidden1_dim,hidden2_dim,layer_name=layer_name,head_num=head_num,layer_num=layer_num)
        self.subgraph_model = Subgraph_GNN(in_dim+random_walk_dim, hidden1_dim,hidden2_dim,k_hop_adj,random_walk_feats,dropout=dropout,random_walk_dim=random_walk_dim,num_layer=subgraph_layer_num)
        self.cls_head = nn.Linear(hidden2_dim*2,num_classes)

    def forward(self,x,edges,walk_feats,hop1,hop2,hop3):
        gnn_output = self.gnn_model(x,edges)
        subgraph_model_out = self.subgraph_model(x,walk_feats,hop1,hop2,hop3)
        h = torch.cat((gnn_output,subgraph_model_out),dim=1)
        return h


class GNN(nn.Module):
    def __init__(self,in_dim, hidden1_dim,hidden2_dim,layer_name='gcn',head_num=16,layer_num = 3):
        super().__init__()
        self.name = layer_name
        self.layer_num = layer_num
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.nn_modules = nn.ModuleList()
        if layer_name=='gcn':
            self.gcn1 = GCNConv(in_dim, hidden1_dim)
            self.gcn2 = GCNConv(hidden1_dim, hidden2_dim)
            if layer_num>2:
                for _ in range(layer_num-2):
                    self.nn_modules.append(GCNConv(hidden2_dim, hidden2_dim))
                    self.nn_modules.append(nn.ReLU())

        elif layer_name=='sage':
            self.gcn1 = SAGEConv(in_dim, hidden1_dim)
            self.gcn2 = SAGEConv(hidden1_dim, hidden2_dim)
            if layer_num > 2:
                for _ in range(layer_num - 2):
                    self.nn_modules.append(SAGEConv(hidden2_dim, hidden2_dim))
                    self.nn_modules.append(nn.ReLU())
        elif layer_name=='gin':
            nn_callable1 = nn.Sequential(nn.Linear(in_dim,hidden1_dim),nn.BatchNorm1d(num_features=hidden1_dim),nn.ReLU(),nn.Linear(hidden1_dim,hidden1_dim),nn.BatchNorm1d(num_features=hidden1_dim))
            nn_callable2 = nn.Sequential(nn.Linear(hidden1_dim,hidden2_dim),nn.BatchNorm1d(num_features=hidden2_dim),nn.ReLU(),nn.Linear(hidden2_dim,hidden2_dim),nn.BatchNorm1d(num_features=hidden2_dim))
            nn_callable3 = nn.Sequential(nn.Linear(hidden2_dim, hidden2_dim), nn.BatchNorm1d(num_features=hidden2_dim),nn.ReLU(),
                                         nn.Linear(hidden2_dim, hidden2_dim), nn.BatchNorm1d(num_features=hidden2_dim))
            nn_callable4 = nn.Sequential(nn.Linear(hidden2_dim, hidden2_dim),nn.BatchNorm1d(num_features=hidden2_dim), nn.ReLU(),
                                         nn.Linear(hidden2_dim, hidden2_dim), nn.BatchNorm1d(num_features=hidden2_dim))
            self.gcn1 = GINConv(nn=nn_callable1)
            self.gcn2 = GINConv(nn=nn_callable2)
            self.gcn3 = GINConv(nn=nn_callable3)
            self.gcn4 = GINConv(nn=nn_callable4)
            self.nn_modules.append(self.gcn3)
            self.nn_modules.append(self.gcn4)
        else:
            print ('gnn module error')


    def forward(self,x,edges):
        if self.name=='gin':
            h = self.gcn1(x,edges)
            h = self.gcn2(h,edges)
            for func in self.nn_modules:
                h = func(h,edges)
            return h
        else:
            h = self.relu1(self.gcn1(x,edges))
            h = self.relu2(self.gcn2(h,edges))
            for func in self.nn_modules:
                if isinstance(func,nn.ReLU):
                    h = func(h)
                else:
                    h = func(h,edges)
            return h


class Subgraph_GNN(nn.Module):
    def __init__(self,in_dim, hidden1_dim,hidden2_dim, k_hop_adj=[0,0,0,0],random_walk_feats=None,dropout = 0.5,random_walk_dim=30,num_layer=2):
        super().__init__()
        self.walk_order = random_walk_dim
        if dropout>0:
            self.layer0 = nn.Sequential(nn.Linear(in_dim,hidden1_dim),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden1_dim,hidden2_dim))
            self.layer1 = nn.Sequential(nn.Linear(in_dim, hidden1_dim), nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden1_dim, hidden2_dim))
            self.layer2 = nn.Sequential(nn.Linear(in_dim, hidden1_dim), nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden1_dim, hidden2_dim))
            self.layer3 = nn.Sequential(nn.Linear(in_dim, hidden1_dim), nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden1_dim, hidden2_dim))
        else:
            if num_layer==2:
                self.layer0 = nn.Sequential(nn.Linear(in_dim,hidden1_dim),nn.BatchNorm1d(hidden1_dim),nn.ReLU(),nn.Linear(hidden1_dim,hidden2_dim))
                self.layer1 = nn.Sequential(nn.Linear(in_dim, hidden1_dim),nn.BatchNorm1d(hidden1_dim),nn.ReLU(),nn.Linear(hidden1_dim, hidden2_dim))
                self.layer2 = nn.Sequential(nn.Linear(in_dim, hidden1_dim),nn.BatchNorm1d(hidden1_dim),nn.ReLU(),nn.Linear(hidden1_dim, hidden2_dim))
                self.layer3 = nn.Sequential(nn.Linear(in_dim, hidden1_dim),nn.BatchNorm1d(hidden1_dim),nn.ReLU(),nn.Linear(hidden1_dim, hidden2_dim))
            else:
                self.layer0 = nn.Sequential(nn.Linear(in_dim,hidden1_dim),nn.BatchNorm1d(hidden1_dim),nn.ReLU(),nn.Linear(hidden1_dim,hidden2_dim),nn.BatchNorm1d(hidden2_dim),nn.ReLU(),nn.Linear(hidden2_dim,hidden2_dim))
                self.layer1 = nn.Sequential(nn.Linear(in_dim,hidden1_dim),nn.BatchNorm1d(hidden1_dim),nn.ReLU(),nn.Linear(hidden1_dim,hidden2_dim),nn.BatchNorm1d(hidden2_dim),nn.ReLU(),nn.Linear(hidden2_dim,hidden2_dim))
                self.layer2 = nn.Sequential(nn.Linear(in_dim,hidden1_dim),nn.BatchNorm1d(hidden1_dim),nn.ReLU(),nn.Linear(hidden1_dim,hidden2_dim),nn.BatchNorm1d(hidden2_dim),nn.ReLU(),nn.Linear(hidden2_dim,hidden2_dim))
                self.layer3 = nn.Sequential(nn.Linear(in_dim,hidden1_dim),nn.BatchNorm1d(hidden1_dim),nn.ReLU(),nn.Linear(hidden1_dim,hidden2_dim),nn.BatchNorm1d(hidden2_dim),nn.ReLU(),nn.Linear(hidden2_dim,hidden2_dim))
            # self.layer0 = nn.Sequential(nn.Linear(in_dim,hidden2_dim))
            # self.layer1 = nn.Sequential(nn.Linear(in_dim, hidden2_dim))
            # self.layer2 = nn.Sequential(nn.Linear(in_dim, hidden2_dim))
            # self.layer3 = nn.Sequential(nn.Linear(in_dim, hidden2_dim))
        # self.hop1,self.hop2,self.hop3 = k_hop_adj[0],k_hop_adj[1],k_hop_adj[2]
        # self.rand = random_walk_feats


    def forward(self, x,walk_feats,hop1,hop2,hop3):
        X = torch.cat((x,walk_feats[:,:self.walk_order]),dim=1)
        hop0_out = self.layer0(X)
        l1_feats = torch.cat((hop1.matmul(x),walk_feats[:,:self.walk_order]),dim=1)
        hop1_out = self.layer1(l1_feats)
        l2_feats = torch.cat((hop2.matmul(x), walk_feats[:, :self.walk_order]),dim=1)
        hop2_out = self.layer2(l2_feats)
        l3_feats = torch.cat((hop3.matmul(x), walk_feats[:, :self.walk_order]),dim=1)
        hop3_out = self.layer3(l3_feats)
        return hop0_out+hop1_out+hop2_out+hop3_out


class PL_GCN(pl.LightningModule):
    def __init__(self,in_dim, hidden1_dim,hidden2_dim,layer_name='gcn',head_num=16,random_walk_dim=10,num_classes=6,lr=1e-2,weight_decay=2e-3,use_benchamark=False,node_classification=False,dropout=0.5,layer_num=3,subgraph_layer_num=2):
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        #         self.save_hyperparameters()
        # Create model
        self.save_hyperparameters()
        self.lr = lr
        self.node_cls = node_classification
        self.weight_decay = weight_decay
        self.benchmark = use_benchamark
        if use_benchamark:
            self.model = GNN(in_dim,hidden1_dim,hidden2_dim,layer_name,head_num,layer_num=layer_num)
            self.cls = nn.Linear(hidden2_dim,num_classes)
        else:
            self.cls = nn.Linear(2*hidden2_dim, num_classes)
            self.args = {'in_dim': in_dim, 'hidden1_dim': hidden1_dim, 'hidden2_dim': hidden2_dim,
                         'layer_name': layer_name, 'random_walk_dim': random_walk_dim, 'num_classes': num_classes,'dropout':dropout,'layer_num':layer_num,'subgraph_layer_num':subgraph_layer_num}
            self.model = Model(**self.args)

        self.log_prob_nn = nn.LogSoftmax(dim=-1)
        self.eval_roc = AUROC(num_classes=num_classes)
        self.acc = Accuracy(top_k=1)


    def init_model(self,k_hop_neibrs,random_walk_feats):
        if not self.benchmark:
            self.model = Model(k_hop_adj=k_hop_neibrs,random_walk_feats=random_walk_feats,**self.args)


    # def set_data(self, pyg_dataset):
    #     self.pyg_fulledge_dataset = pyg_dataset
    #     self.pyg_data = pyg_dataset
    #     self.calc_second_order_adj()


    def forward(self,x,edges):
        # Forward function that is run when visualizing the graph
        h = self.model(x, edges)
        return h

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5,min_lr=1e-3,mode='max',patience=30)
        return {'optimizer':optimizer,'lr_scheduler':scheduler,'monitor':'val_acc'}

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        #         print (batch_idx)
        if not self.benchmark:
            hop1=collate_graph_adj(batch.hop1,batch.ptr,use_cuda)
            hop2 = collate_graph_adj(batch.hop2, batch.ptr,use_cuda)
            hop3 = collate_graph_adj(batch.hop3, batch.ptr,use_cuda)
            h = self.model(batch.x, batch.edge_index,batch.rand_feature,hop1,hop2,hop3)
        else:
            h = self.model(batch.x, batch.edge_index)
        if self.node_cls:
            h=self.log_prob_nn(self.cls(h))
            y = torch.tensor(batch.y).view(-1,)
            loss_val = F.nll_loss(h,y)
        else:
            h = self.cls(h)
            h = self.log_prob_nn(scatter_sum(h,batch.batch,dim=0))
            y = batch.y.view(-1,)
            loss_val = F.nll_loss(h, y)
        self.log("train_loss", loss_val.item(), prog_bar=True, logger=True)

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        #         self.log("train_acc", acc,prog_bar=True, logger=True)
        #         self.log("train_loss", loss,prog_bar=True, logger=True)
        #         self.logger.experiment.add_scalar('tree_em_loss/train',loss.item(),self.global_step)
        return loss_val  # Return tensor to call ".backward" on


    def validation_step(self, batch, batch_idx):
        if not self.benchmark:
            hop1=collate_graph_adj(batch.hop1,batch.ptr,use_gpu=use_cuda)
            hop2 = collate_graph_adj(batch.hop2, batch.ptr,use_cuda)
            hop3 = collate_graph_adj(batch.hop3, batch.ptr,use_cuda)
            h = self.model(batch.x, batch.edge_index,batch.rand_feature,hop1,hop2,hop3)
        else:
            h = self.model(batch.x, batch.edge_index)
        if self.node_cls:
            h=self.log_prob_nn(self.cls(h))
            y = torch.tensor(batch.y).view(-1,)
            loss_val = F.nll_loss(h,y)
        else:
            h = self.cls(h)
            h = self.log_prob_nn(scatter_sum(h,batch.batch,dim=0))
            y = batch.y.view(-1,)
            loss_val = F.nll_loss(h, y)
        acc_val = self.acc(h, y)
        # By default logs it per epoch (weighted average over batches)
        self.log("val_loss", loss_val.item(), prog_bar=True, logger=True)
        self.log("val_acc", acc_val.item(), prog_bar=True, logger=True)
        return acc_val

    def test_step(self, batch, batch_idx):
        if not self.benchmark:
            hop1=collate_graph_adj(batch.hop1,batch.ptr,use_cuda)
            hop2 = collate_graph_adj(batch.hop2, batch.ptr,use_cuda)
            hop3 = collate_graph_adj(batch.hop3, batch.ptr,use_cuda)
            h = self.model(batch.x, batch.edge_index,batch.rand_feature,hop1,hop2,hop3)
        else:
            h = self.model(batch.x, batch.edge_index)
        if self.node_cls:
            h=self.cls(h)
            y = torch.tensor(batch.y).view(-1,)
        else:
            h=self.cls(h)
            h = scatter_sum(h, batch.batch, dim=0)
            y = batch.y.view(-1,)
        acc_val = self.acc(h, y)
        roc_auc = self.eval_roc(h,y)
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log_dict({'acc':acc_val,'rocauc':roc_auc})


if __name__=='__main__':


    # dset = Planetoid(name='CiteSeer', root='data/citeseer/')
    # model = GNN(3703, 16, 15)
    # out = model(dset.data.x, dset.data.edge_index)
    # edges = torch.tensor([[0, 1, 0, 2, 1, 3, 2, 3], [1, 0, 2, 0, 3, 1, 3, 2]]).long()


    # edge_processor = EdgeIndex_Processor(dset.data.edge_index)
    # normed_k_hop_adj, walk_feature = edge_processor.run(random_walk_order=10)
    # num_classes = dset.data.y.max().item() + 1
    # print (dset.data.x.shape)
    # print (len(normed_k_hop_adj))
    # print (walk_feature.shape)


    # run exp of tu dataset
    # run tudataset bench mark
    parser = argparse.ArgumentParser(description='GNN on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--model_name', type=str, default='gcn',
                        help='which method to use')
    parser.add_argument('--benchmark', type=str, default='false',
                        help='whether use benchmark method')
    parser.add_argument('--dataset_name', type=str, default='NCI1',
                        help='which dataset to use')
    parser.add_argument('--layer_num', type=str, default='2,3,4',
                        help='how many layers')
    parser.add_argument('--rw_order', type=str, default='10,16',
                        help='random walk orders')
    parser.add_argument('--num_trial', type=int, default=10,
                        help='how many trials')
    parser.add_argument('--subgraph_layer_num', type=str, default='2,3,4',
                        help='how many hops for the subgraph')
    parser.add_argument('--fname_tag', type=str, default='_trial1',
                        help='suffix to append')
    parser.add_argument('--epoch', type=int, default=150,
                        help='how many epochs to run')
    args = parser.parse_args()


    seeds = np.random.randint(low=1000,high=80000,size=50)
    index = 0
    # datasets_name = ['PTC_FR','NCI1','DD','proteins']
    datasets_name = args.dataset_name
    b = args.benchmark
    b= True if b=='true' else False
    name_tag = args.fname_tag
    max_epochs = args.epoch
    # datasets_name = [args.dataset_name]
    res_all = []
    num_classes_dset = {'proteins':2,'mutag':2,'enzymes':6,'ptc_mr':2}
    layer_name = args.model_name
    layer_num=args.layer_num.split(',')
    layer_num = [float(i) for i in layer_num]
    rw_order = args.rw_order.split(',')
    rw_order = [float(i) for i in rw_order]
    subgraph_layer_num = args.subgraph_layer_num.split(',')
    subgraph_layer_num = [float(i) for i in subgraph_layer_num]
    if b:
        rw_order=[2]
        subgraph_layer_num= [1]

    hiddens = [(32,32),(16,16)]
    for j in range(args.num_trial):
        seed = seeds[index]
        pl.seed_everything(seed)
        index += 1
        num_class = num_classes_dset.get(datasets_name,2)
        random_dim = 11
        dset = TU_Dataset(root=f'data/TUDataset/{datasets_name}_new').shuffle()
        N = len(dset)
        ba = 32
        train_idx = int(0.7 * N)
        valid_idx = int(0.8 * N)
        num_feats = dset[0].x.shape[1]
        dloader_train = DataLoader(dset[:train_idx], batch_size=ba, shuffle=True,num_workers=1)
        dloader_valid = DataLoader(dset[train_idx:valid_idx], batch_size=50, shuffle=False,num_workers=1)
        dloader_test = DataLoader(dset[valid_idx:], batch_size=50, shuffle=False,num_workers=1)

        for ln in layer_num:
            for hidden1,hidden2 in hiddens:
                for sln in subgraph_layer_num:
                    # if b is True and dropout_rate==0.5: continue
                    print ('current_pregress:',datasets_name,j)
                    print (j,datasets_name,b,layer_name,ln,hidden1,hidden2)
                    pl_model = PL_GCN(num_feats, hidden1, hidden2, num_classes=num_class, use_benchamark=b, random_walk_dim=random_dim,node_classification=False,lr=1e-2,layer_name=layer_name,dropout=0.,subgraph_layer_num=sln)
                    trainer = pl.Trainer(default_root_dir=f'saved_models/tudataset/0826_{layer_name}/{datasets_name}/trial_{j}/',gpus=gpu_num, max_epochs=max_epochs,callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")])
                    trainer.fit(model=pl_model, train_dataloaders=dloader_train, val_dataloaders=dloader_valid)
                    model = pl_model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
                    res = trainer.test(model=model,dataloaders=dloader_test)[0]
                    res['name']=datasets_name
                    res['hidden1'] = hidden1
                    res['hidden2'] = hidden2
                    res['seed'] = int(seed)
                    res['model_name'] = layer_name
                    res['benchamark'] = b
                    res['layer_num'] = int(ln)
                    res['subgraph_layer_num'] = int(sln)
                    res_all.append(res)
                    print (f'current result for iter {j}:',res)
    out = json.dumps(res_all)
    with open(f'tudataset_exp/baseline_model_{b}_finished_{datasets_name}_{layer_name}_{name_tag}.json','w') as f:
        f.write(out)
    print (f'write dataset exp data:{datasets_name}_{layer_name}_{name_tag}')
    print (res_all)



