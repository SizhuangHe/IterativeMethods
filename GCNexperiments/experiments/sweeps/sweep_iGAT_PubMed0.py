from __future__ import division
from __future__ import print_function
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
import yaml

import sys
from pathlib import Path
BASE_PATH = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(1, str(BASE_PATH))
import torch
from src.utils.utils import make_Planetoid_data, exp_per_model, make_uniform_schedule
from src.models.iterativeModels import iterativeGAT
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
               notes="Sweep for the iGAT, from greatlakes",
               tags=["iterativeGAT"])
    
    config = wandb.config
    data, num_features, num_classes = make_Planetoid_data(config, seed=2147483647)
    train_schedule = make_uniform_schedule(config.num_iter_layers, config.smooth_fac)
   
    model =iterativeGAT(input_dim=num_features,
                output_dim=num_classes,
                hidden_dim=config.hid_dim,
                train_schedule=train_schedule,
                heads=8,
                dropout=config.dropout,
                attn_dropout_rate=config.dropout,
                xavier_init=True
    )
    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = OneCycleLR(optimizer, max_lr=config.learning_rate, steps_per_epoch=1, epochs=config.num_epochs, pct_start=config.warmup_pct)
    exp_per_model(model, data, optimizer, scheduler,config)

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
        'value': 0.005
    },
    'smooth_fac': {
        'value': 0.6
    },
    'hid_dim': {
        'value': 64
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
        'value': 'PubMed'
    },
    'noise_percent': {
        'value': 0
    },
    'noise_seed':{
        'value': 2147483647
    },
    'warmup_pct':{
        'values': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    }
}
sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project="IterativeMethods")
wandb.agent(sweep_id, run_exp, count=50)
    
        