import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv


class explicit_time_iGCN(nn.Module):
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
        self.gc = GCNConv(hidden_dim + 1, hidden_dim) # one extra dimension of timestep
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
        num_iter = len(schedule)
        current_iter = 1
        for smooth_fac in schedule:      
            old_x = x
            time_stamps = torch.Tensor(np.full(x.shape[0], current_iter/num_iter))
            time_stamps = torch.reshape(time_stamps, (len(time_stamps), 1))
            input_x = torch.cat((x, time_stamps), dim=1)
            x = F.relu(self.gc(input_x, edge_index))
            new_x = F.dropout(x, self.dropout, training=self.training)
            x = self._next_x(old_x, new_x, smooth_fac) 
            current_iter += 1
        x = self.decoder(x)
        return x

class learnable_adaptive_iGCN(nn.Module):
    '''
    This model doesn't take schedule as input. It only takes the number of iterations and smoothing factors are 
    learned as model parameters.
    '''
    def __init__(self, 
                 input_dim, 
                 output_dim,
                 hidden_dim,
                 dropout,
                 num_iterations,
                 eval_schedule=None,
                 xavier_init=False
                 ):
        super().__init__() 
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.gc = GCNConv(hidden_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        # training schedule is now a learnable tensor and initialized to all 0.5's
        self.train_schedule = nn.Parameter(torch.Tensor(np.full(num_iterations, 0.5))) 
        
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
        assert smooth_fac >= 0
        assert smooth_fac <= 1
        next_x = smooth_fac * old_x + (1 - smooth_fac) * new_x
        return next_x

    def forward(self, x, edge_index):
        if self.training:
            schedule = self.train_schedule
        else:
            schedule = self.eval_schedule
        
        print("Schedule: ", schedule)
        
        x = F.relu(self.encoder(x))
        x = F.dropout(x, self.dropout, training=self.training)
        for smooth_fac in F.sigmoid(schedule):   #TODO: may not be correct
            old_x = x
            x = F.relu(self.gc(x, edge_index))
            new_x = F.dropout(x, self.dropout, training=self.training)
            x = self._next_x(old_x, new_x, smooth_fac) 
        x = self.decoder(x)
        return x

class iterativeGCN_variant(nn.Module):
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
        self.encoder = GCNConv(input_dim, hidden_dim)
        self.gc = GCNConv(hidden_dim, hidden_dim)
        self.decoder = GCNConv(hidden_dim, output_dim)
        
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
        
        x = F.relu(self.encoder(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        for smooth_fac in schedule:      
            old_x = x
            x = F.relu(self.gc(x, edge_index))
            new_x = F.dropout(x, self.dropout, training=self.training)
            x = self._next_x(old_x, new_x, smooth_fac) 
        x = self.decoder(x, edge_index)
        return x
    