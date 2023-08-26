import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from src.models.layers import GCNConv_mol, VOCNodeEncoder, GNNInductiveNodeHead
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
# from torch_geometric.utils.dropout import dropout_edge

class GCN(nn.Module):
    '''
    A GCN model with number of layers as input.
    '''
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int,
                 hidden_dim: int,
                 num_layers=2,
                 dropout=0.5):
        super().__init__()
        self.first_gc = GCNConv(input_dim, hidden_dim)
        self.gcs = None
        if num_layers >=3:
            self.gcs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers-2)])
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
        
        return x

class GAT(nn.Module):
    """
    Implementation of Graph Convolutional Networks (https://arxiv.org/pdf/1609.02907.pdf)
    by Kipf and Welling (ICLR 2017).

    Hyperparameters from GAT paper: 2 GAT layers, 8 attention heads computing 8 features each
    (optimized on Cora dataset, used across the rest of their transductive dataset tasks).
    Dropout with p=0.6 on both layers' inputs and on attn coefficients, weight decay 5e-4.

    Intended for node classification tasks.
    """
    def __init__(self, 
                 num_node_features: int, 
                 hidden_dim: int, 
                 output_dim: int,
                 num_layers:int,
                 attn_dropout_rate: float, 
                 dropout: float,
                 heads=8):
        super().__init__()
        self.dropout = dropout

        self.conv1 = GATConv(in_channels=num_node_features, 
                             out_channels=hidden_dim, 
                             heads=heads,
                             dropout=attn_dropout_rate, 
                             concat=True
                             )
        self.act = nn.ELU()
        self.conv2 = GATConv(in_channels=hidden_dim * heads, 
                             out_channels=output_dim, 
                             heads=1,
                             dropout=attn_dropout_rate, 
                             concat=False
                             )
        self.gcs = None
        self.dropout = dropout
        if num_layers >=3:
            self.gcs = nn.ModuleList([GATConv(in_channels=hidden_dim * heads, 
                             out_channels=hidden_dim * heads, 
                             heads=heads,
                             dropout=attn_dropout_rate, 
                             concat=False) for _ in range(num_layers-2)])

    def forward(self, x, edge_index):
        # edge_index = dropout_edge(edge_index=edge_index, p=self.dropout, training=self.training)[0]

        x = self.act(self.conv1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        if self.gcs is not None:
            for layer in self.gcs:
                x = self.act(layer(x, edge_index))
                x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)

        return x

class GCN_arxiv(nn.Module):
    def __init__(self,
                 input_dim:int,
                 output_dim: int,
                 hid_dim:int,
                 num_layers:int,
                 dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.graph_convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        self.graph_convs.append(GCNConv(in_channels=input_dim, out_channels=hid_dim))
        self.batch_norms.append(nn.BatchNorm1d(hid_dim))

        for _ in range(num_layers - 2):
            self.graph_convs.append(GCNConv(in_channels=hid_dim, out_channels=hid_dim))
            self.batch_norms.append(nn.BatchNorm1d(hid_dim))
        
        self.graph_convs.append(GCNConv(in_channels=hid_dim, out_channels=output_dim))
    
    def forward(self, x, adj_t):
        for i, conv in enumerate(self.graph_convs[:-1]):
            x = conv(x, adj_t)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.graph_convs[-1](x, adj_t)
        
        return x
class GCN_mol(nn.Module):
    '''
    This GCN model is for inductive tasks, more specifically, ogbg-molhiv and ogbg-molpcba dataset.
    The code is modified from
      https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/mol/gnn.py
    '''
    def __init__(self, 
                 num_tasks:int,
                 hidden_dim: int,
                 num_layers=2,
                 dropout=0.5):
        '''
        num_tasks (int): number of labels to be predicted
        num_layers (int): number of graph convolutional layers, excluding atom_encoder which encodes the feature to a hidden dimension
        '''
        super().__init__()
        self.num_layers = num_layers
        self.num_tasks = num_tasks
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.atom_encoder = AtomEncoder(hidden_dim)
        
        self.graph_convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        self.pool = global_mean_pool
        self.graph_pred_linear = nn.Linear(self.hidden_dim, self.num_tasks)

        
        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        for layer in range(num_layers):
            self.graph_convs.append(GCNConv_mol(hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))   
        
    def forward(self, x, edge_index, edge_attr, batch):
        h = self.atom_encoder(x)

        for layer in range(self.num_layers):
            h = self.graph_convs[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layers - 1:
                # no ReLu on the last layer
                h = F.dropout(h, self.dropout, training=self.training)
            else:
                h = F.relu(h)
                h = F.dropout(h, self.dropout, training=self.training)

        h = self.pool(h, batch)
        h = self.graph_pred_linear(h)

        return h
    
class GCN_vocsp(nn.Module):
    '''
    This GCN model is for inductive tasks, more specifically, ogbg-molhiv and ogbg-molpcba dataset.
    The code is modified from
      https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/mol/gnn.py
    '''
    def __init__(self, 
                 out_dim:int,
                 hidden_dim: int,
                 MLP_layers=3,
                 num_layers=2,
                 dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.encoder = VOCNodeEncoder(hidden_dim)
        
        self.graph_convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        self.graph_pred_linear = GNNInductiveNodeHead(in_dim=hidden_dim, hid_dim=hidden_dim, out_dim=out_dim, num_layers=MLP_layers)

        
        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        for layer in range(num_layers):
            self.graph_convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))   
        
    def forward(self, x, edge_index, edge_attr, batch):
        h = self.encoder(x)

        for layer in range(self.num_layers):
            h = self.graph_convs[layer](h, edge_index)
            h = self.batch_norms[layer](h)
            h = F.dropout(h, self.dropout, training=self.training)
            h = F.relu(h)
            
        h = self.graph_pred_linear(h)

        return h