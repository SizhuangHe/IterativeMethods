import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from src.models.layers import GCNConv_mol
from ogb.graphproppred.mol_encoder import AtomEncoder

class iterativeGCN(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim,
                 hidden_dim,
                 dropout,
                 train_schedule,
                 eval_schedule=None,
                 xavier_init=False
                 ):
        super().__init__() 
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.gc = GCNConv(hidden_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        
        self.train_schedule = train_schedule
        if eval_schedule is not None:
            self.eval_schedule = eval_schedule
        else:
            self.eval_schedule = self.train_schedule
        self.dropout = dropout
        if xavier_init:
            self._init_xavier()
        
    def _init_xavier(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): # GCNConv layers are already Xavier initilized
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
      
    def _next_x(self, old_x, new_x, smooth_fac):
        next_x = smooth_fac * old_x + (1 - smooth_fac) * new_x
        return next_x

    def forward(self, x, edge_index):
        if self.training:
            schedule = self.train_schedule
        else:
            schedule = self.eval_schedule
        
        x = F.relu(self.encoder(x))
        x = F.dropout(x, self.dropout, training=self.training)
        for smooth_fac in schedule:      
            old_x = x
            x = F.relu(self.gc(x, edge_index))
            new_x = F.dropout(x, self.dropout, training=self.training)
            x = self._next_x(old_x, new_x, smooth_fac) 
        x = self.decoder(x)
        
        return x
    
class iterativeGCN_arxiv(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim,
                 hidden_dim,
                 dropout,
                 train_schedule,
                 eval_schedule=None,
                 xavier_init=False
                 ):
        super().__init__() 
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.gc = GCNConv(hidden_dim, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        
        self.train_schedule = train_schedule
        if eval_schedule is not None:
            self.eval_schedule = eval_schedule
        else:
            self.eval_schedule = self.train_schedule
        self.dropout = dropout
        if xavier_init:
            self._init_xavier()
        
    def _init_xavier(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): # GCNConv layers are already Xavier initilized
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
      
    def _next_x(self, old_x, new_x, smooth_fac):
        next_x = smooth_fac * old_x + (1 - smooth_fac) * new_x
        return next_x

    def forward(self, x, edge_index):
        if self.training:
            schedule = self.train_schedule
        else:
            schedule = self.eval_schedule
        
        x = F.relu(self.encoder(x))
        x = F.dropout(x, self.dropout, training=self.training)
        for smooth_fac in schedule:      
            old_x = x
            x = self.gc(x, edge_index)
            x = self.batch_norm(x)
            x = F.relu(x)
            new_x = F.dropout(x, self.dropout, training=self.training)
            x = self._next_x(old_x, new_x, smooth_fac) 
        x = self.decoder(x)
        
        return x

class iterativeGCN_inductive(nn.Module):
    '''
    iterativeGCN for ogbg-mol datasets
    '''
    def __init__(self,  
                 num_tasks: int,
                 hidden_dim: int,
                 train_schedule,
                 dropout=0.5,
                 eval_schedule=None,
                 xavier_init=False
                 ):
        super().__init__() 
        self.num_tasks = num_tasks
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.train_schedule = train_schedule
        if eval_schedule is not None:
            self.eval_schedule = eval_schedule
        else:
            self.eval_schedule = self.train_schedule

        self.atom_encoder = AtomEncoder(hidden_dim)
        self.graph_conv = GCNConv_mol(hidden_dim, hidden_dim)
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
    
    def forward(self, x, edge_index, edge_attr, batch):
        if self.training:
            schedule = self.train_schedule
        else:
            schedule = self.eval_schedule
        
        x = self.atom_encoder(x)

        for idx in range(len(schedule)):   
            smooth_fac = schedule[idx]    
            old_x = x
            x = self.graph_conv(x, edge_index, edge_attr)
            x = self.batch_norm(x)
            if idx != len(schedule) - 1:
                # no relu on the last iteration
                x = F.relu(x)
            new_x = F.dropout(x, self.dropout, training=self.training)
            x = self._next_x(old_x, new_x, smooth_fac) 
        x = self.pool(x, batch)
        x = self.graph_pred_linear(x)

        return x
    
class iterativeGCN_inductive2(nn.Module):
    '''
    iterativeGCN for ogbg-mol datasets
    '''
    def __init__(self,  
                 num_tasks: int,
                 hidden_dim: int,
                 train_schedule,
                 dropout=0.5,
                 eval_schedule=None,
                 xavier_init=False
                 ):
        super().__init__() 
        self.num_tasks = num_tasks
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.train_schedule = train_schedule
        if eval_schedule is not None:
            self.eval_schedule = eval_schedule
        else:
            self.eval_schedule = self.train_schedule

        self.atom_encoder = AtomEncoder(hidden_dim)
        self.graph_conv = GCNConv_mol(hidden_dim, hidden_dim)
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
    
    def forward(self, x, edge_index, edge_attr, batch):
        if self.training:
            schedule = self.train_schedule
        else:
            schedule = self.eval_schedule
        
        x = self.atom_encoder(x)

        for smooth_fac in range(len(schedule)):      
            old_x = x
            x = self.graph_conv(x, edge_index, edge_attr)
            x = F.relu(x)
            x = self.batch_norm(x)
            new_x = F.dropout(x, self.dropout, training=self.training)
            x = self._next_x(old_x, new_x, smooth_fac) 
        x = self.pool(x, batch)
        x = self.graph_pred_linear(x)

        return x
class iterativeGAT(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int,
                 hidden_dim: int,
                 train_schedule,
                 heads=8,
                 dropout=0.6,
                 attn_dropout_rate=0.6,
                 eval_schedule=None,
                 xavier_init=False
                 ):
        super().__init__() 
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.gc = GATConv(in_channels=hidden_dim, 
                        out_channels=hidden_dim, 
                        heads=heads,
                        dropout=attn_dropout_rate, 
                        concat=False)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        
        self.train_schedule = train_schedule
        if eval_schedule is not None:
            self.eval_schedule = eval_schedule
        else:
            self.eval_schedule = self.train_schedule
        self.dropout = dropout
        if xavier_init:
            self._init_xavier()
        
    def _init_xavier(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): # GCNConv layers are already Xavier initilized
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
      
    def _next_x(self, old_x, new_x, smooth_fac):
        next_x = smooth_fac * old_x + (1 - smooth_fac) * new_x
        return next_x

    def forward(self, x, edge_index):
        if self.training:
            schedule = self.train_schedule
        else:
            schedule = self.eval_schedule
        
        x = F.relu(self.encoder(x))
        x = F.dropout(x, self.dropout, training=self.training)
        for smooth_fac in schedule:      
            old_x = x
            x = F.relu(self.gc(x, edge_index))
            new_x = F.dropout(x, self.dropout, training=self.training)
            x = self._next_x(old_x, new_x, smooth_fac) 
        x = self.decoder(x)
        
        return x