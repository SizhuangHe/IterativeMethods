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

from src.utils.utils import make_Planetoid_data, exp_per_model, make_uniform_schedule
from src.models.auxModels import MLP_GCN

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
               notes="Sweep for the usual GCN",
               tags=["MLPGCN"])
    config = wandb.config
    data, num_features, num_classes = make_Planetoid_data(config)
    model = MLP_GCN(
        input_dim=num_features,
        output_dim=num_classes,
        hidden_dim=config.hid_dim,
        num_layers=config.num_iter_layers,
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
        'values': np.arange(0.0005, 0.02, 0.0005).tolist()
    },
    'smooth_fac': {
        'values': np.arange(0.3, 1, 0.025).tolist()
    },
    'hid_dim': {
        'values': [16, 32]
    },
    'weight_decay': {
        'values': [1e-4, 2e-4, 3e-4, 4e-4, 5e-4]
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
wandb.agent(sweep_id, run_exp, count=50)
    
        