import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import BondEncoder
from torch_geometric.utils import degree
import torch.nn as nn
from torch_geometric.nn import MLP, Linear

class GCNConv_mol(MessagePassing):
    '''
        This GCNConv layer is provided by the Open Graph Benchmark (OGB) team. 
        This is slightly different from the PyG implementation in that it integrates the BondEncoder for ogbg-mol* datasets.
    '''
    def __init__(self, emb_dim):
        super(GCNConv_mol, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

class VOCNodeEncoder(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.encoder = nn.Linear(14, emb_dim)
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, x):
        x = self.encoder(x)
        return x
    
class GNNInductiveNodeHead(nn.Module):
    """
    GNN prediction head for inductive node prediction tasks.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """

    def __init__(self, in_dim, hid_dim, out_dim, num_layers):
        super(GNNInductiveNodeHead, self).__init__()
        layers = []
        if num_layers > 1:
            layers.append(MLP(in_channels=in_dim,
                                 hidden_channels=hid_dim,
                                 out_channels=hid_dim,
                                 num_layers=num_layers - 1,
                                 bias=True))
            layers.append(Linear(in_channels=hid_dim, out_channels=out_dim, bias=True))
        else:
            layers.append(Linear(in_channels=in_dim, out_channels=out_dim, bias=True))

        self.layer_post_mp = nn.Sequential(*layers)
                          
            

    def forward(self, x):
        x = self.layer_post_mp(x)
        return x


if __name__ == "__main__":
    pass