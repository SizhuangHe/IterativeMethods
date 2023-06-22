from __future__ import division
from __future__ import print_function
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from utils import build_iterativeGCN, make_Planetoid_data, exp_per_model
from models import iterativeGCN

import wandb
wandb.login()

def run_exp(config=None):
    wandb.init(job_type="Sweep", 
               project="IterativeMethods", 
               config=config, 
               notes="Fix noise, sweep for the best hyperparams")
    config = wandb.config
    data, num_features, num_classes = make_Planetoid_data(config)
    model = build_iterativeGCN(config, num_features, num_classes)
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
        'value': 6
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
        'values': np.arange(0.3, 0.8, 0.05).tolist()
    }
}
sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project="IterativeMethods")
wandb.agent(sweep_id, run_exp, count=50)
    
        