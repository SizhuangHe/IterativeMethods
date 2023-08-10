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
from src.utils.run_exp import run_PM_iGAT


import wandb
wandb.login()
    

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
wandb.agent(sweep_id, run_PM_iGAT, count=50)
    
        