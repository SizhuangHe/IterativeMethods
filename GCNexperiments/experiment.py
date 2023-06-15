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

DS_NAME = ['Cora', 'CiteSeer', 'PubMed']
# LR = np.arange(0.001, 0.01, 0.0005)
# SMOOTH_FAC = np.arange(0, 1 , 0.1)
WD = np.arange(1e-4, 1e-3, 1e-4)
HID_DIM = [8, 16, 32, 64]

LR = [0.005]
SMOOTH_FAC = [0.5]

for ds_name in DS_NAME:
    dataset = Planetoid(root='data/Planetoid', name=ds_name, transform=NormalizeFeatures())
    data = dataset[0]
    for lr in LR:
        for smooth_fac in SMOOTH_FAC:
            model = iterativeGCN(input_dim=dataset.num_features,
                                    output_dim=dataset.num_classes,
                                    hidden_dim=32,
                                    num_train_iter=2,
                                    num_eval_iter=2,
                                    smooth_fac=smooth_fac,
                                    dropout=0.5)
            loss_test, acc_test, training_time = run_experiment(model=model, 
                                                                data=data, 
                                                                lr=0.0035, 
                                                                weight_decay=2e-4,
                                                                model_name=str(LR) + "," + str(smooth_fac),
                                                                run=1,
                                                                num_epochs=200)
            print("Learning rate: ", lr, ", smoothing factor: ", smooth_fac)
            print("loss: ", loss_test, ", accuracy: ", acc_test)
            del model
