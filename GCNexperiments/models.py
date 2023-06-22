import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv

class only_EncDec(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim,
                 hidden_dim,
                 dropout=0.5,
                 xavier_init=False):
        super().__init__()
        self.first_lin = nn.Linear(input_dim, hidden_dim)
        self.final_lin = nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout
        if xavier_init:
            self._init_xavier()
    
    def _init_xavier(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): 
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index):
        x = F.relu(self.first_lin(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.final_lin(x)
        return F.log_softmax(x, dim=1)

class GCN(nn.Module):
    '''
    A GCN model with number of layers as input. But in general, never do more than 5 layers.
    '''
    def __init__(self, 
                 input_dim, 
                 output_dim,
                 hidden_dim,
                 num_layers=2,
                 dropout=0.5):
        super().__init__()
        self.first_gc = GCNConv(input_dim, hidden_dim)
        self.gcs = None
        if num_layers >=3:
            self.gcs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for i in range(num_layers-2)])
        self.final_gc = GCNConv(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.first_gc(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        if self.gcs is not None:
            for layer in self.gcs:
                x = F.relu(layer(x, edge_index))
                x = F.dropout(x, self.dropout, training=self.training)
        x = self.final_gc(x, edge_index)
        return F.log_softmax(x, dim=1)
    
class MLP_GCN(nn.Module):
    '''
    For experiment. Use same MLP as encoder/decoder and GCs in between.
    '''
    def __init__(self, 
                 input_dim, 
                 output_dim,
                 hidden_dim,
                 num_layers=2,
                 dropout=0.5,
                 xavier_init=False):
        super().__init__()
        self.first_lin = nn.Linear(input_dim, hidden_dim)
        self.gcs = None
        if num_layers >=1:
            self.gcs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for i in range(num_layers)])
        self.final_lin = nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout
        if xavier_init:
            self._init_xavier()

    def _init_xavier(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): 
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index):
        x = F.relu(self.first_lin(x))
        x = F.dropout(x, self.dropout, training=self.training)
        if self.gcs is not None:
            for layer in self.gcs:
                x = F.relu(layer(x, edge_index))
                x = F.dropout(x, self.dropout, training=self.training)
        x = self.final_lin(x)
        return F.log_softmax(x, dim=1)
    
class normalNN(nn.Module):
    '''
    For experiment. Use same MLP as encoder/decoder and GCs in between.
    '''
    def __init__(self, 
                 input_dim, 
                 output_dim,
                 hidden_dim,
                 num_layers=2,
                 dropout=0.5,
                 xavier_init=False):
        super().__init__()
        self.first_lin = nn.Linear(input_dim, hidden_dim)
        self.lins = None
        if num_layers >=1:
            self.lins = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(num_layers)])
        self.final_lin = nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout
        if xavier_init:
            self._init_xavier()
    
    def _init_xavier(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): 
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index):
        x = F.relu(self.first_lin(x))
        x = F.dropout(x, self.dropout, training=self.training)
        if self.lins is not None:
            for layer in self.lins:
                x = F.relu(layer(x))
                x = F.dropout(x, self.dropout, training=self.training)
        x = self.final_lin(x)
        return F.log_softmax(x, dim=1)
    
class iterativeNN(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim,
                 hidden_dim,
                 num_train_iter,
                 smooth_fac,
                 dropout,
                 schedule=None,
                 xavier_init=False
                 ):
        super().__init__() 
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        
        self.train_schedule = np.full(num_train_iter, smooth_fac)
        if schedule is not None:
            self.eval_schedule = schedule
        else:
            self.eval_schedule = self.train_schedule
        
        self.dropout = dropout
        if xavier_init:
            self._init_xavier()
        
    def _init_xavier(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): 
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
            x = F.relu(self.lin(x))
            new_x = F.dropout(x, self.dropout, training=self.training)
            x = self._next_x(old_x, new_x, smooth_fac) 
        x = self.decoder(x)
        return F.log_softmax(x, dim=1)

class iterativeGCN(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim,
                 hidden_dim,
                 num_train_iter,
                 smooth_fac,
                 dropout,
                 schedule=None,
                 xavier_init=False
                 ):
        super().__init__() 
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.gc = GCNConv(hidden_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        
        self.train_schedule = np.full(num_train_iter, smooth_fac)
        if schedule is not None:
            self.eval_schedule = schedule
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
        return F.log_softmax(x, dim=1)
    
class multilayer_iterativeGCN(iterativeGCN):
    def __init__(self, 
                 input_dim, 
                 output_dim,
                 hidden_dim,
                 num_train_iter,
                 smooth_fac,
                 dropout,
                 schedule=None,
                 xavier_init=False,
                 num_layers_per_iteration = 2
                 ):
        self.num_layers_per_iteration = num_layers_per_iteration
        self.encoder = nn.Linear(input_dim, hidden_dim)
        if num_layers_per_iteration < 2:
            raise Exception("invalid number of layers per iteration")
        self.gcs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers_per_iteration)])
        self.decoder = nn.Linear(hidden_dim, output_dim)
        
        self.train_schedule = np.full(num_train_iter, smooth_fac)
        if schedule is not None:
            self.eval_schedule = schedule
        else:
            self.eval_schedule = self.train_schedule
        
        self.dropout = dropout
        if xavier_init:
            self._init_xavier()

        def forward(self, x, edge_index):
            if self.training:
                schedule = self.train_schedule
            else:
                schedule = self.eval_schedule
            
            x = F.relu(self.encoder(x))
            x = F.dropout(x, self.dropout, training=self.training)
            for smooth_fac in schedule:      
                old_x = x
                for layer in self.gcs:
                    x = F.relu(layer(x, edge_index))
                    x = F.dropout(x, self.dropout, training=self.training)
                new_x = x
                x = self._next_x(old_x, new_x, smooth_fac) 
            x = self.decoder(x)
            return F.log_softmax(x, dim=1)