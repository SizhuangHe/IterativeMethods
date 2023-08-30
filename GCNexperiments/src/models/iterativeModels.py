import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from src.models.layers import GCNConv_mol, VOCNodeEncoder, GNNInductiveNodeHead
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.nn import MLP, Linear

class iterativeGCN_Planetoid(nn.Module):
    '''
    This is the iterative version of GCN for the benchmark on the citation datasets.
    It differs from the traditional GCN by iterating over one single Graph Convolutional Layer
      instead of stacking multiple GC layers.
    The node features are encoded into and decoded from a latent space, where iterations over the Graph
     Convolutional layer happen, with MLPs.
    '''
    def __init__(self, 
                 input_dim:int, 
                 output_dim:int,
                 hidden_dim:int,
                 train_schedule,
                 dropout=0.5,
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
    '''
    This iterative version of GCN is for transductive tasks on the ogbg-arxiv dataset. 
    It differs from the version on the Planetoid datasets by a BatchNormalization layer. This is
     inherited from the implementation of the traditional GCN. See this implementation by the OGB team:
     https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/arxiv/gnn.py
    '''
    def __init__(self, 
                 input_dim:int, 
                 output_dim:int,
                 hidden_dim:int,
                 train_schedule,
                 dropout=0.5,
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

class iterativeGCN_mol(nn.Module):
    '''
    This iterative version of GCN is for inductive tasks on the ogbg-mol* datasets.
    Apart from most basic ingredients of iterativeGCNs, it uses:
        - the AtomEncoder provided by the OGB team
        - the BondEncoder provided by the OGB team
        - a slightly different implementation of the GCNConv layer provided by the OGB team
            - It differs from the PyG version by adding BondEncoder to edge_attr
        - a global mean pooling over the batch, since it's doing an inductive task
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
        self.graph_conv = GCNConv_mol(hidden_dim)
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

class iterativeGCN_vocsp(nn.Module):
    def __init__(self,  
                 out_dim: int,
                 hidden_dim: int,
                 train_schedule,
                 MLP_layers=3,
                 dropout=0.5,
                 eval_schedule=None,
                 xavier_init=True
                 ):
        super().__init__() 
        self.out_dim = out_dim
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.train_schedule = train_schedule
        if eval_schedule is not None:
            self.eval_schedule = eval_schedule
        else:
            self.eval_schedule = self.train_schedule

        self.encoder = VOCNodeEncoder(hidden_dim)
        self.graph_conv = GCNConv(hidden_dim, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.graph_pred_linear = GNNInductiveNodeHead(in_dim=hidden_dim, hid_dim=hidden_dim, out_dim=out_dim, num_layers=MLP_layers)
    
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
        
        x = self.encoder(x)

        for smooth_fac in schedule:
            old_x = x
            x = self.graph_conv(x, edge_index)
            x = self.batch_norm(x)
            x = F.relu(x)
            new_x = F.dropout(x, self.dropout, training=self.training)
            x = self._next_x(old_x, new_x, smooth_fac) 
        x = self.graph_pred_linear(x)

        return x
    
class iterativeGCN_peptides(nn.Module):
    def __init__(self,  
                 out_dim: int,
                 hidden_dim: int,
                 train_schedule,
                 MLP_layers=1,
                 dropout=0.5,
                 eval_schedule=None,
                 xavier_init=True
                 ):
        super().__init__() 
        self.out_dim = out_dim
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.train_schedule = train_schedule
        if eval_schedule is not None:
            self.eval_schedule = eval_schedule
        else:
            self.eval_schedule = self.train_schedule

        self.encoder = AtomEncoder(hidden_dim)
        self.graph_conv = GCNConv(hidden_dim, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.pool = global_mean_pool
        self.graph_pred_linear = GNNInductiveNodeHead(in_dim=hidden_dim, hid_dim=hidden_dim, out_dim=out_dim, num_layers=MLP_layers)
    
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
        
        x = self.encoder(x)

        for smooth_fac in schedule:
            old_x = x
            x = self.graph_conv(x, edge_index)
            x = self.batch_norm(x)
            x = F.relu(x)
            new_x = F.dropout(x, self.dropout, training=self.training)
            x = self._next_x(old_x, new_x, smooth_fac) 
        x = self.pool(x, batch)
        x = self.graph_pred_linear(x)

        return x

class iterativeGAT(nn.Module):
    '''
    This is the iterative version of GAT on the Planetoid dataset. It is similar to the iterativeGCN on Planetoid datasets, in that
     the only major difference is GATConv instead of GCNConv.
    '''
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