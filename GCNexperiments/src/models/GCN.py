import torch.nn as nn
import torch.nn.functional as F
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
        return F.log_softmax(x, dim=1)