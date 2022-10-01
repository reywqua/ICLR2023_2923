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

import parser

# # pytorch lightning
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected,add_self_loops,remove_self_loops
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from func_util import *
from ogb_dataset import OGB_Dataset
import argparse

# PyTorch geometric
import torch_geometric
import torch_geometric.data as geo_data
import torch_geometric.nn as geo_nn
from torch_geometric.nn import GCNConv,GATConv,GINConv,SAGEConv
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean,scatter_sum

# ogb
from ogb_modified_code.gnn import GNN
from ogb.graphproppred import PygGraphPropPredDataset,Evaluator


use_cuda = torch.cuda.is_available()
gpu_num = 1 if torch.cuda.is_available() else 0


class Model(nn.Module):
    def __init__(self,emb_dim=300,hidden1_dim=100,hidden2_dim=100,random_walk_dim=10,k_hop_adj=None,random_walk_feats=None,num_classes=6,dropout=0.,num_layer_mhmc = 1):
        super().__init__()
        self.gnn_model = GNN(emb_dim=emb_dim)
        self.subgraph_model = Subgraph_GNN(random_walk_dim, hidden1_dim,hidden2_dim,k_hop_adj,random_walk_feats,dropout=dropout,num_layers=num_layer_mhmc)
        # self.cls_head = nn.Linear(hidden2_dim+emb_dim,num_classes)

    def forward(self,batch,hop1,hop2,hop3):
        gnn_output = self.gnn_model(batch)
        x_encoded = self.gnn_model.gnn_node.x_encoded
        walk_feature = batch.rand_feature
        subgraph_model_out = self.subgraph_model(x_encoded,walk_feature,hop1,hop2,hop3)
        subgraph_model_out = scatter_sum(subgraph_model_out,batch.batch,dim=0)
        h = torch.cat((gnn_output,subgraph_model_out),dim=1)
        return h


class Subgraph_GNN(nn.Module):
    def __init__(self,in_dim, hidden1_dim,hidden2_dim, k_hop_adj=[0,0,0,0],random_walk_feats=None,dropout = 0.,num_layers=1):
        super().__init__()
        self.rw_order = in_dim
        self.num_layers = num_layers

        self.model_dict = nn.ModuleDict()
        for i in range(num_layers):
            if i==0:
                layer0 = nn.Sequential(nn.Linear(in_dim,hidden1_dim),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden1_dim,hidden2_dim))
                layer1 = nn.Sequential(nn.Linear(in_dim, hidden1_dim), nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden1_dim, hidden2_dim))
                layer2 = nn.Sequential(nn.Linear(in_dim, hidden1_dim), nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden1_dim, hidden2_dim))
                layer3 = nn.Sequential(nn.Linear(in_dim, hidden1_dim), nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden1_dim, hidden2_dim))
                self.model_dict[str(i)]= nn.ModuleList([layer0,layer1,layer2,layer3])
            else:
                layer0 = nn.Sequential(nn.Linear(hidden2_dim,hidden1_dim),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden1_dim,hidden2_dim))
                layer1 = nn.Sequential(nn.Linear(hidden2_dim, hidden1_dim), nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden1_dim, hidden2_dim))
                layer2 = nn.Sequential(nn.Linear(hidden2_dim, hidden1_dim), nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden1_dim, hidden2_dim))
                layer3 = nn.Sequential(nn.Linear(hidden2_dim, hidden1_dim), nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden1_dim, hidden2_dim))
                self.model_dict[str(i)]= nn.ModuleList([layer0,layer1,layer2,layer3])

        # self.hop1,self.hop2,self.hop3 = k_hop_adj[0],k_hop_adj[1],k_hop_adj[2]
        # self.rand = random_walk_feats


    def forward(self, x,walk_feats,hop1,hop2,hop3):
        # X = torch.cat((x,walk_feats),dim=1)
        X= walk_feats[:,:self.rw_order]
        for i in range(self.num_layers):
            i = str(i)
            layer0,layer1,layer2,layer3 = self.model_dict[i][0],self.model_dict[i][1],self.model_dict[i][2],self.model_dict[i][3]
            hop0_out = layer0(X)
            hop1_out = layer1(hop1.matmul(X))
            hop2_out = layer2(hop2.matmul(X))
            hop3_out = layer3(hop3.matmul(X))
            X = hop0_out+hop1_out+hop2_out+hop3_out
        return X

class PL_GCN(pl.LightningModule):
    def __init__(self,emb_dim, hidden1_dim=100,hidden2_dim=100,random_walk_dim=10,num_classes=6,lr=1e-3,use_benchamark=False,node_classification=False,dropout=0.5,num_layer_mhmc = 2):
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        #         self.save_hyperparameters()
        # Create model
        self.save_hyperparameters()
        self.lr = lr
        self.node_cls = node_classification
        self.benchmark = use_benchamark
        if use_benchamark:
            print ('benchmark not in this script')
            self.model = GNN(emb_dim=emb_dim,num_tasks=num_classes)
            # self.cls = nn.Linear(hidden2_dim,num_classes)
        else:
            self.cls = nn.Linear(hidden2_dim+emb_dim, num_classes)
            self.args = {'emb_dim': emb_dim, 'hidden1_dim': hidden1_dim, 'hidden2_dim': hidden2_dim,
                        'random_walk_dim': random_walk_dim, 'num_classes': num_classes,'dropout':dropout,'num_layer_mhmc':num_layer_mhmc}
            self.model = Model(**self.args)

        self.loss_func = nn.BCEWithLogitsLoss()
        self.eval_roc = AUROC(num_classes=num_classes)
        self.acc = Accuracy(top_k=1)


    def init_model(self,k_hop_neibrs,random_walk_feats):
        if not self.benchmark:
            self.model = Model(k_hop_adj=k_hop_neibrs,random_walk_feats=random_walk_feats,**self.args)


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

    def forward(self,batch):
        # Forward function that is run when visualizing the graph
        h = self.model(batch)
        return h

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5,min_lr=5e-4,mode='max',patience=40)
        return {'optimizer':optimizer}

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        #         print (batch_idx)
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            is_labeled = batch.y == batch.y
            if not self.benchmark:
                hop1= self.collate_graph_adj(batch.hop1,batch.ptr)
                hop2 = self.collate_graph_adj(batch.hop2, batch.ptr)
                hop3 = self.collate_graph_adj(batch.hop3, batch.ptr)
                h = self.model(batch,hop1,hop2,hop3)
                pred = self.cls(h)
                pred = pred[is_labeled]
                y = batch.y[is_labeled].float()
                loss_val = self.loss_func(pred,y)
            else:
                return
            self.log("train_loss", loss_val.item(), prog_bar=True, logger=True)

            # Logs the accuracy per epoch to tensorboard (weighted average over batches)
            #         self.log("train_acc", acc,prog_bar=True, logger=True)
            #         self.log("train_loss", loss,prog_bar=True, logger=True)
            #         self.logger.experiment.add_scalar('tree_em_loss/train',loss.item(),self.global_step)
            return loss_val  # Return tensor to call ".backward" on


    def validation_epoch_end(self, outputs):
        val_data = outputs[0]
        test_data = outputs[1]
        y_true_val = np.vstack([i[0] for i in val_data])
        y_pred_val = np.vstack([i[1] for i in val_data])
        y_true_test = np.vstack([i[0] for i in test_data])
        y_pred_test = np.vstack([i[1] for i in test_data])
        input_dict = {"y_true": y_true_val, "y_pred": y_pred_val}
        val_ap = evaluator.eval(input_dict)['rocauc']
        input_dict = {"y_true": y_true_test, "y_pred": y_pred_test}
        test_ap = evaluator.eval(input_dict)['rocauc']
        exp_data.append([val_ap,test_ap])
        res = {'val_rocauc':val_ap,'test_rocauc':test_ap}
        self.log_dict(res)
        print (f'current epoch:{self.current_epoch},result:{res}\n')

    def validation_step(self, batch, batch_idx,dataloader_idx):
        y_true = []
        y_pred = []
        log_dict = {0:'val_auc',1:'test_auc'}
        if batch.x.shape[0]==1:
            pass
        else:
            if not self.benchmark:
                hop1= self.collate_graph_adj(batch.hop1,batch.ptr)
                hop2 = self.collate_graph_adj(batch.hop2, batch.ptr)
                hop3 = self.collate_graph_adj(batch.hop3, batch.ptr)
                h = self.model(batch,hop1,hop2,hop3)
                pred = self.cls(h)
                y_true.append(batch.y.view(pred.shape).detach().cpu())
                y_pred.append(pred.detach().cpu())
                y_true = torch.cat(y_true, dim=0).numpy()
                y_pred = torch.cat(y_pred, dim=0).numpy()
                return [y_true,y_pred]
                # input_dict = {"y_true": y_true, "y_pred": y_pred}
                # val = evaluator.eval(input_dict)['rocauc']
                # # By default logs it per epoch (weighted average over batches)
                # if dataloader_idx == 0:
                #     self.log(log_dict[0], val, prog_bar=True, logger=True)
                # else:
                #     self.log(log_dict[1], val, prog_bar=True, logger=True)
            else:
                return

    def test_step(self, batch, batch_idx):
        y_true = []
        y_pred = []
        if not self.benchmark:
            hop1=collate_graph_adj(batch.hop1,batch.ptr,use_cuda)
            hop2 = collate_graph_adj(batch.hop2, batch.ptr,use_cuda)
            hop3 = collate_graph_adj(batch.hop3, batch.ptr,use_cuda)
            pred = self.cls(self.model(batch,hop1,hop2,hop3))
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())
            y_true = torch.cat(y_true, dim=0).numpy()
            y_pred = torch.cat(y_pred, dim=0).numpy()
            input_dict = {"y_true": y_true, "y_pred": y_pred}
            val = evaluator.eval(input_dict)['rocauc']
            self.log("test_rocauc", val, prog_bar=True, logger=True)
        else:
            pred = self.model(batch)
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())
            y_true = torch.cat(y_true, dim=0).numpy()
            y_pred = torch.cat(y_pred, dim=0).numpy()
            input_dict = {"y_true": y_true, "y_pred": y_pred}
            val = evaluator.eval(input_dict)['rocauc']
        # By default logs it per epoch (weighted average over batches)
            self.log("test_rocauc", val, prog_bar=True, logger=True)
        # By default logs it per epoch (weighted average over batches), and returns it afterwards


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


    # args = {'in_dim':3703,'hidden1_dim':50,'hidden2_dim':20,'layer_name':'gcn','random_walk_dim':10,'k_hop_adj':normed_k_hop_adj,'random_walk_feats':walk_feature,'num_classes':num_classes}
    # model = Model(**args)
    # # model = Subgraph_GNN(3703+10+1,64,32,normed_k_hop_adj,walk_feature,4)
    # x,edges = dset.data.x,dset.data.edge_index
    # q = model(x,edges)
    # print (q.shape)

    # dset = Synthetic_Dataset(root='data/pyg_TRIANGLE_EX/train')   # TRIANGLE:classes=2   LCC:classes=3
    # # for d in dset:
    # #     if max(d.y)>1:
    # #         print (d.y)
    # dset_test = Synthetic_Dataset(root='data/pyg_TRIANGLE_EX/test')
    # dloader_train = DataLoader(dset[:900],batch_size=50,shuffle=True)
    # dloader_valid = DataLoader(dset[900:], batch_size=50, shuffle=False)
    # dloader_test = DataLoader(dset_test, batch_size=100, shuffle=False)
    #
    # pl_model = PL_GCN(5,15,12,num_classes=2,use_benchamark=False,random_walk_dim=10)
    # trainer = pl.Trainer(gpus=0,max_epochs=100,callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=10)])
    # trainer.fit(model=pl_model,train_dataloaders=dloader_train,val_dataloaders=dloader_valid)
    # trainer.test(model=pl_model,dataloaders=dloader_test)
    parser = argparse.ArgumentParser(description='GNN on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--num_layer_mhmc', type=int, default=2,
                        help='whether use benchmark method')

    parser.add_argument('--hidden_dims', type=int, default=32,
                        help='whether use benchmark method')


    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='whether use benchmark method')
    args = parser.parse_args()


    # run exp of tu dataset
    # run tudataset bench mark
    exp_data = []
    evaluator = Evaluator('ogbg-molhiv')
    res_all = []
    dataset = PygGraphPropPredDataset(name='ogbg-molhiv', root='data/ogb/')
    split_idx = dataset.get_idx_split()
    dataset = OGB_Dataset(root='data/ogb/'+'molhiv')
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=50, shuffle=True)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=100, shuffle=False)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=100, shuffle=False)
    num_class=1
    num_layers_mhmc = args.num_layer_mhmc
    hidden_num = args.hidden_dims
    dr_rate = args.dropout_rate
    for j in range(5):
        seed = np.random.randint(1000, 60000)
        pl.seed_everything(seed)
        for b in [False]:
            for r in [20]:
                exp_data = []
                # if b is True and dropout_rate==0.5: continue
                pl_model = PL_GCN(emb_dim=290,num_classes=num_class, use_benchamark=b, random_walk_dim=r,node_classification=False,lr=1e-3,dropout=dr_rate,hidden1_dim=hidden_num,hidden2_dim=hidden_num,num_layer_mhmc=num_layers_mhmc)
                trainer = pl.Trainer(auto_select_gpus=True,default_root_dir=f'saved_models/ogb_molhiv/',gpus=gpu_num, max_epochs=120)
                trainer.fit(model=pl_model, train_dataloaders=train_loader, val_dataloaders=[valid_loader,test_loader])
                # model = pl_model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
                # res = trainer.test(model=model,dataloaders=test_loader)[0]
                # res['seed'] = seed
                # res['benchamark'] = b
                # res_all.append(res)

                top1 = sorted(exp_data,key = lambda x:x[0],reverse=True)[0][1]
                top3 = sorted(exp_data, key=lambda x: x[0], reverse=True)[:3]
                top3 = np.asarray([i[1] for i in top3]).mean()
                print (f'iter_num:{j},random_order:{r},result for top1:{top1},top3:{top3}')

