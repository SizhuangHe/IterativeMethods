from __future__ import division
from __future__ import print_function
import numpy as np
import torch

import sys
from pathlib import Path
BASE_PATH = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(1, str(BASE_PATH))
import torch

from src.utils.metrics import MAD
from src.models.models import GCN_inductive
from src.utils.utils import exp_mol
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.loader import DataLoader

import wandb
wandb.login()

'''
This script is for sweeping for a set of hyperparameters for the usual GCN,
on the Cora dataset with a fixed amount of noise.
'''

def run_exp(config=None):
    wandb.init(job_type="molpcba",
               project="IterativeMethods", 
               config=config, 
               notes="usualGCN",
               tags=["usualGCN"])
    config = wandb.config
    dataset = PygGraphPropPredDataset(name="ogbg-molpcba") 
    split_idx = dataset.get_idx_split() 
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False)
    evaluator = Evaluator(name="ogbg-molpcba")
    model = GCN_inductive(
            num_tasks=dataset.num_tasks,
            hidden_dim=config.hid_dim,
            num_layers=config.num_iter_layers,
            dropout=0.5)
    exp_mol(model, train_loader, valid_loader, test_loader, evaluator, config)
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
        'values': [2,3,4,5,6,7,8,9]
    },
    'learning_rate': {
        'values': np.arange(0.001, 0.02, 0.0005).tolist()
    },
    'smooth_fac': {
        'value': 0.5 # does't matter
    },
    'hid_dim': {
        'values': [16, 32]
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
        'value': 'CiteSeer'
    },
    'noise_percent': {
        'value': 0
    },
    'noise_seed':{
        'value': 2147483647
    }
}
sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project="IterativeMethods")
wandb.agent(sweep_id, run_exp, count=50)
    
        