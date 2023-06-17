import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv

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
        # self.gcs = None
        # if num_layers >=3:
        #     self.gcs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for i in range(num_layers-2)])
        self.final_gc = GCNConv(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.first_gc(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        # if self.gcs is not None:
        #     for layer in self.gcs:
        #         x = F.relu(layer(x, edge_index))
        #         x = F.dropout(x, self.dropout, training=self.training)
        x = self.final_gc(x, edge_index)
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
        
        self.num_train_iter = num_train_iter
        self.schedule = schedule
        self.smooth_fac = smooth_fac
        self.dropout = dropout
        if xavier_init:
            self._initialize_weights()
        

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                # GCNConv layers are already Xavier initilized
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    
    def _next_x(self, old_x, new_x, smooth_fac):
        next_x = smooth_fac * old_x + (1 - smooth_fac) * new_x
        return next_x

    def forward(self, x, edge_index):
        x = F.relu(self.encoder(x))
        x = F.dropout(x, self.dropout, training=self.training)
        
        schedule = np.full(self.num_train_iter, self.smooth_fac)
        if (not self.training) and (self.schedule is not None):
            schedule = self.schedule

        for iter in range(len(schedule)):
            smooth_fac = schedule[iter]
            old_x = x
            x = F.relu(self.gc(x, edge_index))
            new_x = F.dropout(x, self.dropout, training=self.training)
            x = self._next_x(old_x, new_x, smooth_fac)
            
        x = self.decoder(x)
        return F.log_softmax(x, dim=1)