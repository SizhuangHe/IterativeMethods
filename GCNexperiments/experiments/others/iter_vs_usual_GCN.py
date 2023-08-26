from __future__ import division
from __future__ import print_function
import numpy as np
import torch
import torch.nn.functional as F

import sys
from pathlib import Path
BASE_PATH = Path(__file__).parent.parent.absolute()
sys.path.insert(1, str(BASE_PATH))

from src.utils.utils import make_Planetoid_data, exp_per_model, make_uniform_schedule, build_iterativeGCN
from src.models.GCN import GCN
from src.models.iterativeModels import iterativeGCN, iterativeGCN_variant
import wandb
wandb.login()

'''
In this script, we first add noise (a fixed percentage of random 1's) to the dataset and then evaluate
the performance of two architectures found in sweeps earlier.
'''

def run_exp(config_iter=None, config_usual=None):
    data, num_features, num_classes = make_Planetoid_data(config_iter)

    wandb.init(config=config_iter,
               job_type="model_compare",
               project="IterativeMethods",
               notes="Compare iterativeGCN and usual GCN on the same noisy dataset",
               tags=["iterativeGCN"],
               reinit=True
    )
    config = wandb.config
    train_schedule = make_uniform_schedule(config.num_iter_layers, config.smooth_fac)
    iterative_gcn = build_iterativeGCN(config, num_features, num_classes, train_schedule)
    exp_per_model(iterative_gcn, data, config)
    del iterative_gcn
    wandb.finish()

    wandb.init(config=config_usual,
               job_type="model_compare",
               project="IterativeMethods",
               notes="Compare iterativeGCN and usual GCN on the same noisy dataset",
               tags=["usualGCN"]
    )
    config = wandb.config
    gcn = GCN(input_dim=num_features,
              output_dim=num_features,
              hidden_dim=config.hid_dim,
              num_layers=config.num_iter_layers,
              dropout=config.dropout)
    exp_per_model(gcn, data, config)
    del gcn
    wandb.finish()

    wandb.init(config=config_iter_var,
               job_type="model_compare",
               project="IterativeMethods",
               notes="Compare iterativeGCN and usual GCN on the same noisy dataset",
               tags=["iterativeGCNvariant"]
    )
    config = wandb.config
    train_schedule = make_uniform_schedule(config.num_iter_layers, config.smooth_fac)
    gcn_var = iterativeGCN_variant(input_dim=num_features,
                                 output_dim=num_classes,
                                 hidden_dim=config.hid_dim,
                                 train_schedule=train_schedule,
                                 dropout=config.dropout,
                                 xavier_init=True
                                 )
    exp_per_model(gcn_var, data, config)
    del gcn_var
    wandb.finish()


    


config_iter = {
    'num_epochs': 200,
    'dataset_name': "Cora",
    'noise_percent': 0.5,
    'hid_dim': 32,
    'num_iter_layers': 7,
    'smooth_fac': 0.6,
    'dropout': 0.5,
    'learning_rate': 0.0035,
    'weight_decay': 3e-4
} 

config_usual = {
    'num_epochs': 200,
    'dataset_name': "Cora",
    'noise_percent': 0.7,
    'hid_dim': 32,
    'num_iter_layers': 4,
    'smooth_fac': 0.45,
    'dropout': 0.5,
    'learning_rate': 0.012,
    'weight_decay': 5e-4
} 

config_iter_var = {
    'num_epochs': 200,
    'dataset_name': "Cora",
    'noise_percent': 0.5,
    'hid_dim': 32,
    'num_iter_layers': 9,
    'smooth_fac': 0.7,
    'dropout': 0.5,
    'learning_rate': 0.004,
    'weight_decay': 5e-4
} 

run_exp(config_iter, config_usual)