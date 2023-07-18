import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
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
                             concat=True)
        self.act = nn.ELU()
        self.conv2 = GATConv(in_channels=hidden_dim * heads, 
                             out_channels=output_dim, 
                             heads=1,
                             dropout=attn_dropout_rate, 
                             concat=False)
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