from __future__ import division
from __future__ import print_function
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from utils import build_iterativeGCN, make_Planetoid_data, exp_per_model, make_uniform_schedule
from models import iterativeGCN_variant

import wandb
wandb.login()

'''
This script is to sweep for the best set of hyperparameters for iterativeGCN,
given Cora dataset with a fixed amount of noise.
'''

def run_exp(config=None):
    wandb.init(job_type="Sweep", 
               project="IterativeMethods", 
               config=config, 
               notes="variant of iGCN experiments",
               tags=["iterativeGCNvariant"])
    config = wandb.config
    train_schedule = make_uniform_schedule(config.num_iter_layers, config.smooth_fac)
    wandb.log({
        "train_schedule": train_schedule
    })
    data, num_features, num_classes = make_Planetoid_data(config)
    model = iterativeGCN_variant(input_dim=num_features,
                                 output_dim=num_classes,
                                 hidden_dim=config.hid_dim,
                                 train_schedule=train_schedule,
                                 dropout=config.dropout,
                                 xavier_init=True
                                 )
    exp_per_model(model, data, config)
    wandb.finish()
    

config = {
    'num_epochs': 200,
    'dataset_name': "Cora",
    'noise_percent': 0,
    'hid_dim': 32,
    'num_iter_layers': 2,
    'smooth_fac': 0.7,
    'dropout': 0.5,
    'learning_rate': 0.01,
    'weight_decay': 5e-4
} 
run_exp(config)
        