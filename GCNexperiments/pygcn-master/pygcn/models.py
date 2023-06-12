import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import time

class GCN_2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_2, self).__init__()
        '''
        added by Sizhuang: 
            
        - This model is a 2-layer GCN.
            - First layer:    nfeat to nhid
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

        print("Intialize a 2-layer GCN")
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        # x: input data matrix
        # adj: adjacency matrix
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj) 
        return F.log_softmax(x, dim=1)

class GCN_5(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_5, self).__init__()
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
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, nhid)
        self.gc5 = GraphConvolution(nhid, nclass)
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
    def __init__(self, nfeat, nclass, dropout, train_nite, eval_nite=0, allow_grad=True, smooth_fac=0):
        '''     
        - This model is a 1-layer GCN with nite iterations, followed by a linear layer and a log_softmax
            - GC layer:     nfeat to nfeat
            - linear layer: nfeat to nclass, (to cast hidden representations of nodes to a dimension of nclass)
        - Activation: ReLu
        - Input:
            - nfeat:        the number of features of each node
            - nclass:       the number of target classes (we are doing a node classification task here)
            - dropout:      dropout rate
            - train_nite:   the number of iterations during training
            - eval_nite:    the number of iterations during evaluation, 
                            if not specified (or invalid), intialize to the same as train_nite
            - allow_grad:   (bool) defaulted to True. 
                            whether or nor allow gradients to flow through all GC iterations, 
                            if False, gradients will only flow to the last iteration
            - smooth_fac:   a number in [0,1], smoothing factor, controls how much of the OLD iteration result is
                            counted in the skip connection in each iteration
                            for example, smooth_fac = x means y_{i+1} = x * y_i + (1-x) * y_{i+1}
                            Invalid inputs will be treated as 0.
        - Output:
            - A probability vector of length nclass, by log_softmax
        '''
        super(ite_GCN, self).__init__()

        self.gc = GraphConvolution(nfeat, nfeat)
        self.linear_no_bias = nn.Linear(nfeat, nclass, bias=False)
        self.dropout = dropout
        self.train_nite = train_nite
        self.allow_grad = allow_grad
        self.smooth_fac = smooth_fac
        self.eval_nite = eval_nite
        
        if (smooth_fac > 1) or (smooth_fac < 0):
            print("Invalid smoothing factor. Treat as 0.")
            self.smooth_fac = 0
        if (eval_nite <= 0):
            print("Unspecified or invalid number of iterations for inference. Treat as the same as training iterations.")
            self.eval_nite = self.train_nite
        
        print("Initialize a 1-layer GCN with ", self.train_nite, "iterations")
        print("Gradient flows to all iterations: ", allow_grad)

    def run_one_layer(self, x, adj):
        x_old = x
        x_new = self.gc(x, adj)
        x = F.relu(self.smooth_fac * x_old + (1 - self.smooth_fac) * x_new)
        x = F.dropout(x, self.dropout, training=self.training)
        return x

    def forward(self, x, adj):
        if self.training:
            for i in range(self.train_nite):
                if not self.allow_grad:
                    x = x.detach()
                    x = self.run_one_layer(x, adj)
                else:
                    x = self.run_one_layer(x, adj)
        else:
            for i in range(self.eval_nite):
                x = self.run_one_layer(x, adj)

        x = self.linear_no_bias(x)
        return F.log_softmax(x, dim=1)
        