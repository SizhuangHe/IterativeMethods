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
        'value': 5
    },
    'learning_rate': {
        'value': 0.001
    },
    'smooth_fac': {
        'value': 0.5 # does't matter
    },
    'hid_dim': {
        'value': 300
    },
    'weight_decay': {
        'value': 0
    },
    'num_epochs': {
        'value': 100
    },
    'dropout': {
        'value': 0.6
    },
    'dataset_name': {
        'value': 'ogbg-molhiv'
    },
    'noise_percent': {
        'value': 0
    },
    'noise_seed':{
        'value': 2147483647
    }
}
sweep_config['parameters'] = parameters_dict

'''
This script is for sweeping for a set of hyperparameters for the usual GCN,
on the Cora dataset with a fixed amount of noise.
'''
dataset = PygGraphPropPredDataset(name='ogbg-molhiv') 
split_idx = dataset.get_idx_split() 
train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True)
valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False)
test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False)
evaluator = Evaluator(name="ogbg-molhiv")

def run_exp(config=None):
    wandb.init(job_type="molhiv",
               project="IterativeMethods", 
               config=config, 
               notes="usualGCN",
               tags=["usualGCN"])
    config = wandb.config
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    wandb.log({
        "device": device_str
    })

    model = GCN_inductive(
            num_tasks=dataset.num_tasks,
            hidden_dim=config.hid_dim,
            num_layers=config.num_iter_layers,
            dropout=0.5)
    model = model.to(device)
    exp_mol(model, train_loader, valid_loader, test_loader, evaluator, config, device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    wandb.finish()
    
        


sweep_id = wandb.sweep(sweep_config, project="IterativeMethods")
wandb.agent(sweep_id, run_exp, count=100)
    
        