from __future__ import division
from __future__ import print_function
import numpy as np
import torch

import sys
from pathlib import Path
BASE_PATH = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(1, str(BASE_PATH))
import torch
from src.utils.utils import make_Planetoid_data, exp_per_model, make_uniform_schedule
from src.models.models import GAT
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
               notes="Sweep for the GAT, from greatlakes",
               tags=["GAT"])
    config = wandb.config
    data, num_features, num_classes = make_Planetoid_data(config, seed=2147483647)
    train_schedule = make_uniform_schedule(config.num_iter_layers, config.smooth_fac)
    model = GAT(num_node_features=num_features,
                hidden_dim=config.hid_dim,
                output_dim=num_classes,
                num_layers=config.num_iter_layers,
                attn_dropout_rate=config.dropout,
                dropout=config.dropout,
                heads=8)
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
        'value': 0.0035
    },
    'smooth_fac': {
        'value': 0.5 # does't matter
    },
    'hid_dim': {
        'value': 8
    },
    'weight_decay': {
        'value': 5e-4
    },
    'num_epochs': {
        'value': 200
    },
    'dropout': {
        'value': 0.6
    },
    'dataset_name': {
        'value': 'Cora'
    },
    'noise_percent': {
        'value': 0.5
    },
    'noise_seed':{
        'value': 2147483647
    }
}
sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project="IterativeMethods")
wandb.agent(sweep_id, run_exp, count=100)
    
        