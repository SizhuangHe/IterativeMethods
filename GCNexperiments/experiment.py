from __future__ import division
from __future__ import print_function

import time
import logging
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric

from  utils import run_experiment
from models import GCN, iterativeGCN
from torch_geometric.datasets import Planetoid

from torch_geometric.transforms import NormalizeFeatures

dataset = Planetoid(root='data/Planetoid', name="Cora", transform=NormalizeFeatures())
data = dataset[0]

lr = 0.0085
smooth_fac = 0.8
num_runs = 20

total_acc = 0
total_time = 0
print("Iterative: ")
for run in range(num_runs):
    model = iterativeGCN(input_dim=dataset.num_features,
                            output_dim=dataset.num_classes,
                            hidden_dim=32,
                            num_train_iter=2,
                            num_eval_iter=2,
                            smooth_fac=smooth_fac,
                            dropout=0.5)
    loss_test, acc_test, training_time = run_experiment(model=model, 
                                                        data=data, 
                                                        lr=lr, 
                                                        weight_decay=2e-4,
                                                        model_name=str(lr) + "_" + str(smooth_fac),
                                                        run=1,
                                                        num_epochs=200,
                                                        )
    total_acc += acc_test
    total_time += training_time

    del model
mean_acc = total_acc / num_runs
mean_time = total_time / num_runs
print("Mean accuracy: ", mean_acc, " mean training time: ", mean_time)

# Ran hidden dimension=16  200 times and mean accuracy is 0.745
# Ran hidden dimension=32  20 times and mean accuracy is 0.77
print("=================")
print("Non-iterative")
total_acc = 0
total_time = 0
for run in range(num_runs):
    model = GCN(input_dim=dataset.num_features,
                output_dim=dataset.num_classes,
                hidden_dim=32,
                num_layers=2,
                dropout=0.5)
    loss_test, acc_test, training_time = run_experiment(model=model, 
                                                        data=data, 
                                                        lr=lr, 
                                                        weight_decay=2e-4,
                                                        model_name=str(lr) + "_" + str(smooth_fac),
                                                        run=1,
                                                        num_epochs=200,
                                                        )
    total_acc += acc_test
    total_time += training_time

    del model
mean_acc = total_acc / num_runs
mean_time = total_time / num_runs
print("Mean accuracy: ", mean_acc, " mean training time: ", mean_time)

# Ran hidden dimension=32 20 times, mean accuracy=0.814