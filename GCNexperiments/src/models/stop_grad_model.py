import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

class iterativeGCN_inductive(nn.Module):
    '''
    iterativeGCN for ogbg-mol datasets
    This is used to try only compute gradients on the last iteration
    '''
    def __init__(self,  
                 num_tasks: int,
                 hidden_dim: int,
                 train_schedule,
                 dropout=0.5,
                 eval_schedule=None,
                 xavier_init=False,
                 stop_grad = False
                 ):
        super().__init__() 
        self.num_tasks = num_tasks
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.stop_grad = stop_grad
        self.train_schedule = train_schedule
        if eval_schedule is not None:
            self.eval_schedule = eval_schedule
        else:
            self.eval_schedule = self.train_schedule

        self.atom_encoder = AtomEncoder(hidden_dim)
        self.graph_conv = GCNConv(hidden_dim, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.pool = global_mean_pool
        self.graph_pred_linear = nn.Linear(self.hidden_dim, self.num_tasks)
    
    def _init_xavier(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): # GCNConv layers are already Xavier initilized
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
      
    def _next_x(self, old_x, new_x, smooth_fac):
        next_x = smooth_fac * old_x + (1 - smooth_fac) * new_x
        return next_x
    
    def forward(self, x, edge_index, batch):
        if self.training:
            schedule = self.train_schedule
        else:
            schedule = self.eval_schedule
        
        x = self.atom_encoder(x)

        for smooth_fac in schedule:
            if self.stop_grad:
                x = x.detach()
            old_x = x
            x = self.graph_conv(x, edge_index)
            x = F.relu(x)
            x = self.batch_norm(x)
            new_x = F.dropout(x, self.dropout, training=self.training)
            x = self._next_x(old_x, new_x, smooth_fac) 
        
        x = self.pool(x, batch)
        x = self.graph_pred_linear(x)

        return x
