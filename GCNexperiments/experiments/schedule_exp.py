from __future__ import division
from __future__ import print_function
import time
import numpy as np
import copy
import torch
import logging
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

import sys
from pathlib import Path
BASE_PATH = Path(__file__).parent.parent.absolute()
sys.path.insert(1, str(BASE_PATH))

from src.utils.utils import test
from src.models.GCN import GCN
from src.models.iterativeModels import iterativeGCN

'''
This script is not up to date.
'''


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s: %(message)s")
file_handler = logging.FileHandler("schedule_exp.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

sigmoid = F.sigmoid(torch.Tensor(np.linspace(1, 9, 30)))
uniform = np.full(30, 0.97)
linear = np.linspace(0.9, 1, 30)
tanh = F.tanh(torch.Tensor(np.linspace(1, 5, 30)))

lr = 0.0075
weight_decay = 5e-4
num_runs = 100
smooth_fac = 0.7
hid_dim = 32
num_iter = 2

dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
data = dataset[0]

ACC_orig = []
ACC_sigm = []
ACC_unif = []
ACC_line = []
ACC_tanh = []

for i in range(num_runs):
    start_t = time.time()
    model_origin = iterativeGCN(input_dim=dataset.num_features,
                                output_dim=dataset.num_classes,
                                hidden_dim=hid_dim,
                                num_train_iter=num_iter,
                                smooth_fac=smooth_fac,
                                schedule=None,
                                dropout=0.5,
                                xavier_init=True
                                )
    loss_test, acc_test, training_time = run_experiment(model=model_origin,
                                                        data=data,
                                                        lr=lr,
                                                        weight_decay=weight_decay,
                                                        model_name="lala",
                                                        run=1,
                                                        num_epochs=200
                                                        ) 
    ACC_orig.append(acc_test)
    orig_state_dict = copy.deepcopy(model_origin.state_dict())
    del model_origin
    
    model_sigm = iterativeGCN(input_dim=dataset.num_features,
                                output_dim=dataset.num_classes,
                                hidden_dim=hid_dim,
                                num_train_iter=num_iter,
                                smooth_fac=smooth_fac,
                                schedule=sigmoid,
                                dropout=0.5,
                                xavier_init=True
                                )
    model_sigm.load_state_dict(orig_state_dict)
    loss_test, acc_test = test(model=model_sigm, data=data)
    ACC_sigm.append(acc_test)
    del model_sigm

    model_unif = iterativeGCN(input_dim=dataset.num_features,
                                output_dim=dataset.num_classes,
                                hidden_dim=hid_dim,
                                num_train_iter=num_iter,
                                smooth_fac=smooth_fac,
                                schedule=uniform,
                                dropout=0.5,
                                xavier_init=True
                                )
    model_unif.load_state_dict(orig_state_dict)
    loss_test, acc_test = test(model=model_unif, data=data)
    ACC_unif.append(acc_test)
    del model_unif

    model_line = iterativeGCN(input_dim=dataset.num_features,
                                output_dim=dataset.num_classes,
                                hidden_dim=hid_dim,
                                num_train_iter=num_iter,
                                smooth_fac=smooth_fac,
                                schedule=linear,
                                dropout=0.5,
                                xavier_init=True
                                )
    model_line.load_state_dict(orig_state_dict)
    loss_test, acc_test = test(model=model_line, data=data)
    ACC_line.append(acc_test)
    del model_line

    model_tanh = iterativeGCN(input_dim=dataset.num_features,
                                output_dim=dataset.num_classes,
                                hidden_dim=hid_dim,
                                num_train_iter=num_iter,
                                smooth_fac=smooth_fac,
                                schedule=tanh,
                                dropout=0.5,
                                xavier_init=True
                                )
    model_tanh.load_state_dict(orig_state_dict)
    loss_test, acc_test = test(model=model_tanh, data=data)
    ACC_tanh.append(acc_test)
    del model_tanh
    
    end_t = time.time()
    print("Run {:03d}/{:03d}, accuracy: no schedule {:.4}, sigmoid {:.4}, uniform {:.4}, linear {:.4}, tanh {:.4}, time elapsed {:.4}".format(i+1, num_runs, ACC_orig[-1], ACC_sigm[-1], ACC_unif[-1], ACC_line[-1], ACC_tanh[-1], end_t-start_t))

fig = plt.figure(figsize =(10, 7))
plt.violinplot([ACC_orig, ACC_sigm, ACC_unif, ACC_line, ACC_tanh], showmedians=True, showextrema=True)
fig.savefig("schedule_exp")

mean_orig = np.mean(ACC_orig)
mean_sigm = np.mean(ACC_sigm)
mean_unif = np.mean(ACC_unif)
mean_line = np.mean(ACC_line)
mean_tanh = np.mean(ACC_tanh)
std_orig = np.std(ACC_orig)
std_sigm = np.std(ACC_sigm)
std_unif = np.std(ACC_unif)
std_line = np.std(ACC_line)
std_tanh = np.std(ACC_tanh)

logger.info("--> Experiment: lr={:.4}, weight_decay={:.4}, smooth_fac={:.4}, hidden_dim={:02d}, num_iteration={}".format(lr, weight_decay, smooth_fac, hid_dim, num_iter))
logger.info("No schedule: {}".format(np.full(num_iter, smooth_fac)))
logger.info("Sigmoid schedule: {}".format(sigmoid))
logger.info("Uniform schedule: {}".format(uniform))
logger.info("Linear schedule: {}".format(linear))
logger.info("Tanh schedule: {}".format(tanh))
logger.info("Result:\n no schedule: {:.4} +/- {:.4},\n sigmoid: {:.4} +/- {:.4},\n uniform: {:.4} +/- {:.4},\n linear: {:.4} +/- {:.4},\n tanh: {:.4} +/- {:.4}".format(mean_orig, std_orig, mean_sigm, std_sigm, mean_unif, std_unif, mean_line, std_line, mean_tanh, std_tanh))