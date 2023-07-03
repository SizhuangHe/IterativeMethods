from __future__ import division
from __future__ import print_function
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

import sys
from pathlib import Path
BASE_PATH = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(1, str(BASE_PATH))

from src.utils.utils import build_iterativeGCN, make_Planetoid_data, exp_per_model, make_uniform_schedule
from src.models.iterativeModels import iterativeGCN_variant

import wandb
from wandb import AlertLevel
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

sweep_config = {
    'method': 'random'
}

metric = {
    'name': 'accuracy',
    'goal': 'maximize'
}
sweep_config['metric'] = metric

parameters_dict = {
    'num_iter_layers': {
        'values': [2, 3, 4, 5, 6, 7, 8, 9]
    },
    'learning_rate': {
        'value': 0.003
    },
    'smooth_fac': {
        'value': 0.65
    },
    'hid_dim': {
        'value': 32
    },
    'weight_decay': {
        'value': 2e-4
    },
    'num_epochs': {
        'value': 200
    },
    'dropout': {
        'value': 0.5
    },
    'dataset_name': {
        'value': 'Cora'
    },
    'noise_percent': {
        'value': 0.5
    }
}
sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project="IterativeMethods")
wandb.agent(sweep_id, run_exp, count=400)
    
        