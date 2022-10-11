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
from torch_geometric.nn import GCNConv,GATConv,GINConv,GATv2Conv,SAGEConv
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean,scatter_sum
import scipy.io as sio
from torchmetrics import MeanSquaredError
import os
from libs.spect_conv import SpectConv,ML3Layer
from libs.utils import SRDataset,SpectralDesign

from libs.spect_conv import SpectConv,ML3Layer
from libs.utils import PlanarSATPairsDataset,SpectralDesign
from libs.spect_conv import SpectConv,ML3Layer
from libs.utils import GraphCountDataset,SpectralDesign,get_n_params


from pyg_dataset import TU_Dataset
from wl_test_model_utils import *
from torch_geometric.transforms import Compose
import torch_geometric.transforms as T

import argparse


os.environ["CUDA_VISIBLE_DEVICES"] = '3'

class PPGN(nn.Module):
    def __init__(self, nmax=37, nneuron=16,hidden2=64):
        super(PPGN, self).__init__()

        self.nmax = nmax
        self.nneuron = nneuron
        ninp = 27

        bias = False
        self.mlp1_1 = torch.nn.Conv2d(ninp, nneuron, 1, bias=bias)
        self.mlp1_2 = torch.nn.Conv2d(ninp, nneuron, 1, bias=bias)
        self.mlp1_3 = torch.nn.Conv2d(nneuron + ninp, nneuron, 1, bias=bias)

        self.mlp2_1 = torch.nn.Conv2d(nneuron, nneuron, 1, bias=bias)
        self.mlp2_2 = torch.nn.Conv2d(nneuron, nneuron, 1, bias=bias)
        self.mlp2_3 = torch.nn.Conv2d(2 * nneuron, nneuron, 1, bias=bias)

        self.mlp3_1 = torch.nn.Conv2d(nneuron, nneuron, 1, bias=bias)
        self.mlp3_2 = torch.nn.Conv2d(nneuron, nneuron, 1, bias=bias)
        self.mlp3_3 = torch.nn.Conv2d(2 * nneuron, nneuron, 1, bias=bias)

        self.mlp4_1 = torch.nn.Conv2d(nneuron, nneuron, 1, bias=bias)
        self.mlp4_2 = torch.nn.Conv2d(nneuron, nneuron, 1, bias=bias)
        self.mlp4_3 = torch.nn.Conv2d(2 * nneuron, nneuron, 1, bias=bias)

        self.h1 = torch.nn.Linear(2 * 4 * nneuron, hidden2)
        # self.h2 = torch.nn.Linear(64, 1)

    def forward(self, data):
        x = data.X2
        M = torch.sum(data.M, (1), True)

        x1 = F.relu(self.mlp1_1(x) * M)
        x2 = F.relu(self.mlp1_2(x) * M)
        x1x2 = torch.matmul(x1, x2) * M
        x = F.relu(self.mlp1_3(torch.cat([x1x2, x], 1)) * M)

        # sum layer readout
        xo1 = torch.cat([torch.sum(x * data.M[:, 0:1, :, :], (2, 3)), torch.sum(x * data.M[:, 1:2, :, :], (2, 3))], 1)

        x1 = F.relu(self.mlp2_1(x) * M)
        x2 = F.relu(self.mlp2_2(x) * M)
        x1x2 = torch.matmul(x1, x2) * M
        x = F.relu(self.mlp2_3(torch.cat([x1x2, x], 1)) * M)

        # sum layer readout
        xo2 = torch.cat([torch.sum(x * data.M[:, 0:1, :, :], (2, 3)), torch.sum(x * data.M[:, 1:2, :, :], (2, 3))], 1)

        x1 = F.relu(self.mlp3_1(x) * M)
        x2 = F.relu(self.mlp3_2(x) * M)
        x1x2 = torch.matmul(x1, x2) * M
        x = F.relu(self.mlp3_3(torch.cat([x1x2, x], 1)) * M)

        # sum  layer readout
        xo3 = torch.cat([torch.sum(x * data.M[:, 0:1, :, :], (2, 3)), torch.sum(x * data.M[:, 1:2, :, :], (2, 3))], 1)

        x1 = F.relu(self.mlp4_1(x) * M)
        x2 = F.relu(self.mlp4_2(x) * M)
        x1x2 = torch.matmul(x1, x2) * M
        x = F.relu(self.mlp4_3(torch.cat([x1x2, x], 1)) * M)

        # sum  layer readout
        xo4 = torch.cat([torch.sum(x * data.M[:, 0:1, :, :], (2, 3)), torch.sum(x * data.M[:, 1:2, :, :], (2, 3))], 1)

        x = torch.cat([xo1, xo2, xo3, xo4], 1)
        x = F.relu(self.h1(x))
        return x


class Model(nn.Module):
    def __init__(self,in_dim, hidden1_dim,hidden2_dim,layer_name='ppgn',head_num=16,random_walk_dim=5,k_hop_adj=None,random_walk_feats=None,num_classes=6,use_base_gnn = False,use_random_walk = True,layer_num=2):
        super().__init__()
        if layer_name=='ppgn':
            print ('using ppgn................')
            self.gnn_model = PPGN(hidden2=hidden2_dim)
        else:
            self.gnn_model = GNN(in_dim, hidden1_dim,hidden2_dim,layer_name=layer_name,head_num=head_num)
        if use_random_walk:
            self.subgraph_model = Subgraph_GNN(in_dim+random_walk_dim+1, hidden1_dim,hidden2_dim,k_hop_adj,random_walk_feats,use_rw= use_random_walk,layer_num = layer_num)
        else:
            self.subgraph_model = Subgraph_GNN(in_dim, hidden1_dim, hidden2_dim, k_hop_adj, random_walk_feats, use_rw=use_random_walk,layer_num=layer_num)
        # self.cls_head = nn.Linear(hidden2_dim*2,num_classes)
        self.use_rw = use_random_walk
        self.use_gnn = use_base_gnn

    def forward(self,x,edges,batch):
        if self.use_gnn:
            gnn_output = self.gnn_model(batch)

        subgraph_model_out = self.subgraph_model(batch)
        subgraph_model_out = scatter_sum(subgraph_model_out, batch.batch, dim=0)
        if self.use_gnn:
            h = torch.cat((gnn_output,subgraph_model_out*0.0),dim=1)
            return h
        else:
            return subgraph_model_out



class Subgraph_GNN(nn.Module):
    def __init__(self,in_dim, hidden1_dim,hidden2_dim, k_hop_adj=[0,0,0,0],random_walk_feats=None,use_rw = True,layer_num=2):
        super().__init__()
        if layer_num==2:
            self.layer0 = nn.Sequential(nn.Linear(in_dim,hidden1_dim),nn.ReLU(),nn.Dropout(0.5),nn.Linear(hidden1_dim,hidden2_dim))
            self.layer1 = nn.Sequential(nn.Linear(in_dim, hidden1_dim), nn.ReLU(),nn.Dropout(0.5), nn.Linear(hidden1_dim, hidden2_dim))
            self.layer2 = nn.Sequential(nn.Linear(in_dim, hidden1_dim), nn.ReLU(),nn.Dropout(0.5), nn.Linear(hidden1_dim, hidden2_dim))
            self.layer3 = nn.Sequential(nn.Linear(in_dim, hidden1_dim), nn.ReLU(),nn.Dropout(0.5), nn.Linear(hidden1_dim, hidden2_dim))
        else:
            self.layer0 = nn.Sequential(nn.Linear(in_dim,hidden1_dim),nn.ReLU(),nn.Dropout(0.5),nn.Linear(hidden1_dim,hidden2_dim),nn.ReLU(),nn.Dropout(0.5),nn.Linear(hidden2_dim,hidden2_dim))
            self.layer1 = nn.Sequential(nn.Linear(in_dim,hidden1_dim),nn.ReLU(),nn.Dropout(0.5),nn.Linear(hidden1_dim,hidden2_dim),nn.ReLU(),nn.Dropout(0.5),nn.Linear(hidden2_dim,hidden2_dim))
            self.layer2 = nn.Sequential(nn.Linear(in_dim,hidden1_dim),nn.ReLU(),nn.Dropout(0.5),nn.Linear(hidden1_dim,hidden2_dim),nn.ReLU(),nn.Dropout(0.5),nn.Linear(hidden2_dim,hidden2_dim))
            self.layer3 = nn.Sequential(nn.Linear(in_dim,hidden1_dim),nn.ReLU(),nn.Dropout(0.5),nn.Linear(hidden1_dim,hidden2_dim),nn.ReLU(),nn.Dropout(0.5),nn.Linear(hidden2_dim,hidden2_dim))
        # self.hop1,self.hop2,self.hop3 = k_hop_adj[0],k_hop_adj[1],k_hop_adj[2]
        # self.rand = random_walk_feats
        self.use_rw = use_rw


    def forward(self,batch):
        X = torch.cat((batch.x,batch.rand_feature),dim=1)
        hop1_f = batch.hop1_feature
        hop2_f = batch.hop2_feature
        hop3_f = batch.hop3_feature
        hop0_out = self.layer0(X)
        hop1_out = self.layer1(hop1_f)
        hop2_out = self.layer2(hop2_f)
        hop3_out = self.layer3(hop3_f)
        return hop0_out+hop1_out+hop2_out+hop3_out


class PL_GCN(pl.LightningModule):
    def __init__(self,in_dim, hidden1_dim,hidden2_dim,layer_name='gcn',head_num=16,random_walk_dim=10,num_classes=6,lr=1e-2,weight_decay=2e-3,use_benchmark=False,node_classification=False,use_gnn=False,use_rw=True,layer_num=2):
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        #         self.save_hyperparameters()
        # Create model
        self.save_hyperparameters()
        self.lr = lr
        self.node_cls = node_classification
        self.weight_decay = weight_decay
        self.benchmark = use_benchmark
        self.mse = MeanSquaredError()
        if use_benchmark:
            self.model = PPGN(hidden2=hidden2_dim)
            self.cls = nn.Linear(hidden2_dim,1)
        else:
            if use_gnn:
                self.cls = nn.Linear(2*hidden2_dim, 1)
            else:
                self.cls = nn.Linear(hidden2_dim, 1)
            self.args = {'in_dim': in_dim, 'hidden1_dim': hidden1_dim, 'hidden2_dim': hidden2_dim,
                         'layer_name': layer_name, 'random_walk_dim': random_walk_dim, 'num_classes': num_classes,'use_base_gnn':use_gnn,'use_random_walk':use_rw,'layer_num':layer_num}
            self.model = Model(**self.args)

        self.log_prob_nn = nn.LogSoftmax(dim=-1)



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
            # hop1=self.collate_graph_adj(batch.hop1,batch.ptr)
            # hop2 = self.collate_graph_adj(batch.hop2, batch.ptr)
            # hop3 = self.collate_graph_adj(batch.hop3, batch.ptr)
            h = self.model(batch.x, batch.edge_index,batch)
        else:
            h = self.model(batch)
        h=self.cls(h).view(-1,)
        y = batch.y.view(-1,)
        loss_val = F.l1_loss(h,y)
        self.log("train_loss", loss_val.item(), prog_bar=True)

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        #         self.log("train_acc", acc,prog_bar=True, logger=True)
        #         self.log("train_loss", loss,prog_bar=True, logger=True)
        #         self.logger.experiment.add_scalar('tree_em_loss/train',loss.item(),self.global_step)
        return loss_val  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        if not self.benchmark:
            h = self.model(batch.x, batch.edge_index,batch)
        else:
            h = self.model(batch)

        h=self.cls(h).view(-1,)
        y = batch.y.view(-1,)
        loss_val = F.l1_loss(h,y)
        # By default logs it per epoch (weighted average over batches)
        self.log("val_loss", loss_val.item(), prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        if not self.benchmark:
            h = self.model(batch.x, batch.edge_index,batch)
        else:
            h = self.model(batch)
        h=self.cls(h).view(-1,)
        y = batch.y.view(-1,)
        loss_val = F.l1_loss(h,y)
        # By default logs it per epoch (weighted average over batches)
        self.log("test_loss", loss_val.item(), prog_bar=True, logger=True)



if __name__=='__main__':
    # dset = Planetoid(name='CiteSeer', root='data/citeseer/')
    # model = GNN(3703, 16, 15)
    # out = model(dset.data.x, dset.data.edge_index)
    # edges = torch.tensor([[0, 1, 0, 2, 1, 3, 2, 3], [1, 0, 2, 0, 3, 1, 3, 2]]).long()

    parser = argparse.ArgumentParser(description='GNN on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--use_gnn', type=int, default=1,
                        help='whether use benchmark method')

    parser.add_argument('--benchmark', type=int, default=0,
                        help='whether use benchmark method')

    parser.add_argument('--layer_name', type=str, default='ppgn',
                        help='whether use benchmark method')

    parser.add_argument('--layer_num', type=int, default=2,
                        help='whether use benchmark method')

    args = parser.parse_args()

    results = []
    # transform = SpectralDesign(nmax=30, recfield=1, dv=1, nfreq=10, adddegree=True, laplacien=False, addadj=True)
    # transforms = Compose([transform, T.OneHotDegree(max_degree=6), Norm_y()])
    dataset = TU_Dataset(root='dataset/ZINC_new/')
    print (dataset[0])
    trid = list(range(0, 10000))
    vlid = list(range(10000, 11000))
    tsid = list(range(11000, 12000))
    use_benchmark=True if args.benchmark==1 else False
    use_gnn = True if args.use_gnn == 1 else False
    layer_name = args.layer_name
    layer_num = args.layer_num

    for i in range(5):
        seed = np.random.randint(5000, 60000)
        pl.seed_everything(seed)

        train_loader = DataLoader(dataset[trid], batch_size=64, shuffle=True,num_workers=1)
        val_loader = DataLoader(dataset[vlid], batch_size=64, shuffle=False,num_workers=1)
        test_loader = DataLoader(dataset[tsid], batch_size=64, shuffle=False,num_workers=1)

        num_feats = dataset[0].x.shape[1]
        hidden1 = 64
        hidden2 = 64
        use_gnn = True if args.use_gnn==1 else False
        pl_model = PL_GCN(num_feats, hidden1, hidden2, num_classes=1, use_benchmark=use_benchmark, random_walk_dim=20,node_classification=False, lr=1e-3, layer_name=layer_name,use_gnn=use_gnn,use_rw=True,layer_num=layer_num)
        trainer = pl.Trainer(default_root_dir=f'saved_models/zinc/0820_ppgn/', gpus=1 if torch.cuda.is_available() else 0, max_epochs=300,
                             callbacks=[EarlyStopping(patience=800, monitor='val_loss', mode='min'),
                                        ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss")])
        trainer.fit(model=pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        model = pl_model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        res = trainer.test(model=model, dataloaders=test_loader)[0]
        print ()
        print (f'test result for iter{i}:{res}')
        print ()
        res['iteration'] = i
        results.append(res)
    print (results)


