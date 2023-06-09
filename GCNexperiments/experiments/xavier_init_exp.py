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

import sys
from pathlib import Path
BASE_PATH = Path(__file__).parent.parent.absolute()
sys.path.insert(1, str(BASE_PATH))


from src.models.GCN import GCN
from src.models.iterativeModels import iterativeGCN
from torch_geometric.datasets import Planetoid

from torch_geometric.transforms import NormalizeFeatures

'''
This script is not up to date.
'''

dataset = Planetoid(root='data/Planetoid', name="Cora", transform=NormalizeFeatures())
data = dataset[0]

lr = 0.0085
smooth_fac = 0.8
num_runs = 50
hidden_dim = 32
train_iter = 2
num_layers = 2
weight_decay = 2e-4
num_epochs = 200

total_acc = 0
total_time = 0

print("Total runs: {}, Estimated running time: {} s.".format(num_runs*2, num_runs*6))

for run in range(num_runs):
    model = iterativeGCN(input_dim=dataset.num_features,
                            output_dim=dataset.num_classes,
                            hidden_dim=hidden_dim,
                            num_train_iter=train_iter,
                            num_eval_iter=train_iter,
                            smooth_fac=smooth_fac,
                            dropout=0.5)
    loss_test, acc_test, training_time = run_experiment(model=model, 
                                                        data=data, 
                                                        lr=lr, 
                                                        weight_decay=weight_decay,
                                                        model_name=str(lr) + "_" + str(smooth_fac),
                                                        run=run,
                                                        num_epochs=num_epochs,
                                                        )
    total_acc += acc_test
    total_time += training_time
    print("Run {:03d}/{:03d} finished, time elapsed: {:.2f}s".format(run+1, num_runs, time.time()-t_start))
    del model
mean_acc = total_acc / num_runs
mean_time = total_time / num_runs