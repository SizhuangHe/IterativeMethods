import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import time

class GCN_3(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_3, self).__init__()
        '''
        added by Sizhuang: 
            
        - This model is a 3-layer GCN.
            - First layer:    nfeat to nhid
            - Second layer:   nhid to nhid
            - Final layer:    nhid to nclass
        - Activation: ReLu
        - Input:
            - nfeat: the number of features of each node
            - nhid: the dimension of the hidden representation for each node
            - nclass: the number of target classes (we are doing a node classification task here)
            - dropout: dropout rate
        - Output:
            - A probability vector of length nclass, by log_softmax
        '''

        print("Intialize a 3-layer GCN")
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        # x: input data matrix
        # adj: adjacency matrix
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        return F.log_softmax(x, dim=1)

class ite_GCN(nn.Module):
    def __init__(self, nfeat, nclass, dropout, nite, allow_grad=True):
        '''     
        - This model is a 1-layer GCN with nite iterations, followed by a linear layer and a log_softmax
            - GC layer:     nfeat to nfeat
            - linear layer: nfeat to nclass, (to cast hidden representations of nodes to a dimension of nclass)
        - Activation: ReLu
        - Input:
            - nfeat:        the number of features of each node
            - nclass:       the number of target classes (we are doing a node classification task here)
            - dropout:      dropout rate
            - nite:         the number of iterations of the GC layer
            - allow_grad:   (bool) defaulted to True. 
                            whether or nor allow gradients to flow through all GC iterations, 
                            if False, gradients will only flow to the last iteration
        - Output:
            - A probability vector of length nclass, by log_softmax
        '''
        super(ite_GCN, self).__init__()

        self.gc = GraphConvolution(nfeat, nfeat)
        self.linear_no_bias = nn.Linear(nfeat, nclass)
        self.dropout = dropout
        self.nite = nite
        self.allow_grad = allow_grad
        print("Initialize a 1-layer GCN with ", self.nite, "iterations")
        print("Gradient flows to all iterations: ", allow_grad)

    def forward(self, x, adj):
        
        for i in range(self.nite - 1): 
            # all iterations except the last one may not require gradients
            x = F.relu(self.gc(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
            
            if not self.allow_grad:
                x.detach()
        
        x = F.relu(self.gc(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.linear_no_bias(x)
        return F.log_softmax(x, dim=1)
        