from __future__ import division
from __future__ import print_function
import numpy as np
import torch

import sys
from pathlib import Path
BASE_PATH = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(1, str(BASE_PATH))
import torch
from src.utils.utils import make_Planetoid_data, exp_per_model
from src.models.models import GCN
from src.utils.metrics import MAD

import wandb
wandb.login()

'''
This script is for sweeping for a set of hyperparameters for the usual GCN,
on the Cora dataset with a fixed amount of noise.
'''

def run_exp(config=None):
    wandb.init(job_type="Sweep", 
               project="IterativeMethods", 
               config=config, 
               notes="Sweep for the usual GCN, from greatlakes",
               tags=["usualGCN"])
    config = wandb.config
    data, num_features, num_classes = make_Planetoid_data(config, seed=2147483647)
    model = GCN(
        input_dim=num_features,
        output_dim=num_classes,
        hidden_dim=config.hid_dim,
        num_layers=config.num_iter_layers,
        dropout=config.dropout
    )
    exp_per_model(model, data, config)

    out = model(data.x, data.edge_index)
    mad = MAD(out.detach())
    wandb.log({
        "MAD": mad
    })

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
        'value': 2
    },
    'learning_rate': {
        'value': 0.01
    },
    'smooth_fac': {
        'value': 0.5 # does't matter
    },
    'hid_dim': {
        'value': 32
    },
    'weight_decay': {
        'value': 5e-4
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
        'value': 0.7
    },
    'noise_seed':{
        'value': 2147483647
    }
}
sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project="IterativeMethods")
wandb.agent(sweep_id, run_exp, count=100)
    
        