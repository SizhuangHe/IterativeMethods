from __future__ import division
from __future__ import print_function

import time
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from  utils import run_experiment
from models import GCN, iterativeGCN
from torch_geometric.datasets import Planetoid

from torch_geometric.transforms import NormalizeFeatures

schedule = [0.3, 0.3, 0.3]

dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
data = dataset[0]

model = iterativeGCN(input_dim=dataset.num_features,
                                        output_dim=dataset.num_classes,
                                        hidden_dim=16,
                                        num_train_iter=2,
                                        smooth_fac=0.7,
                                        schedule=schedule,
                                        dropout=0.5)

from utils import accuracy
lr = 0.01
weight_decay = 5e-4
def run_experiment(data, lr, weight_decay, model_name, run, num_epochs=200, plot_fig=False):
    loss_TRAIN = []
    acc_TRAIN = []
    loss_VAL = []
    acc_VAL = []

    optimizer = optim.Adam(model.parameters(),
                       lr=lr, weight_decay=weight_decay)
    total_start = time.time()
    for epoch in range(num_epochs):
        t = time.time()
    
        model.train()

        output = model(data.x, data.edge_index)   
        loss_train = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
        pred = output[data.train_mask].argmax(dim=1)
        acc_train = accuracy(pred, data.y[data.train_mask])
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

    total_end = time.time()
    training_time = total_end - total_start
    

run_experiment(data=data,
               lr=lr,
               weight_decay=weight_decay,
               model_name="lala",
               run=1)

model.eval()
start_time = time.time()
output = model(data.x, data.edge_index)
end_time = time.time()

loss_test = F.nll_loss(output[data.test_mask], data.y[data.test_mask])
pred = output[data.test_mask].argmax(dim=1)
acc_test = accuracy(pred, data.y[data.test_mask])
    
print(acc_test)