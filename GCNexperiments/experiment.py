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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s: %(message)s")
file_handler = logging.FileHandler("experiment.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

dataset = Planetoid(root='data/Planetoid', name="Cora", transform=NormalizeFeatures())
data = dataset[0]

logger.info("Experiment begins.")
logger.info("Dataset used: Cora")

lr = 0.0085
smooth_fac = 0.8
num_runs = 70
hidden_dim = 32
train_iter = 2
num_layers = 2
weight_decay = 2e-4
num_epochs = 200

total_acc = 0
total_time = 0

logger.info("Iterative model info:\n Hidden dimension: {:d}\n training iterations: {:d}\n smoothing factor: {:.2f}".format(hidden_dim, train_iter, smooth_fac))
logger.info("Experiment info: learning rate: {:.4f}\n weight_decay: {:.2e}\n number of epochs: {:d}".format(lr, weight_decay, num_epochs))
t_start = time.time()
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
logger.info("Ran iterative models {} times, Mean accuracy: {:.4}, mean training time: {:.4} s".format(num_runs, mean_acc, mean_time))


total_acc = 0
total_time = 0
lr - 0.01
logger.info("Non-terative model info:\n Hidden dimension: {:d}\n number of layers: {:d}\n".format(hidden_dim, train_iter))
logger.info("Experiment info: learning rate: {:.4f}\n weight_decay: {:.2e}\n number of epochs: {:d}".format(lr, weight_decay, num_epochs))
t_start = time.time()
for run in range(num_runs):
    model = GCN(input_dim=dataset.num_features,
                output_dim=dataset.num_classes,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
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
    print("Run {:03d}/{:03d} finished, time elapsed: {:.2f} s".format(run+1, num_runs, time.time()-t_start))
    del model
mean_acc = total_acc / num_runs
mean_time = total_time / num_runs
logger.info("Ran Non-iterative models {} times, Mean accuracy: {:.4}, mean training time: {:.4} s".format(num_runs, mean_acc, mean_time))
