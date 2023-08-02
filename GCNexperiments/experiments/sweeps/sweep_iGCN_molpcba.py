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
from src.models.iterativeModels import iterativeGCN_inductive
from src.utils.utils import exp_mol, make_uniform_schedule
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import Adam

import wandb
wandb.login()


sweep_config = {
    'method': 'random'
}

metric = {
    'name': 'rocauc',
    'goal': 'maximize'
}
sweep_config['metric'] = metric

parameters_dict = {
    'num_iter_layers': {
        'values': [4,5,6,7,8,9,10]
    },
    'learning_rate': {
        'values': np.arange(0.0005, 0.002, 0.0001).tolist()
    },
    'smooth_fac': {
        'values': np.arange(0.1, 0.95, 0.05).tolist()
    },
    'hid_dim': {
        'values': [300, 350, 400, 450, 500, 550, 600]
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
        'value': 'ogbg-molpcba'
    },
    'warmup_pct': {
        'values': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    }
}
sweep_config['parameters'] = parameters_dict

'''
This script is for sweeping for a set of hyperparameters for the usual GCN,
on the Cora dataset with a fixed amount of noise.
'''
dataset = PygGraphPropPredDataset(name='ogbg-molpcba') 
split_idx = dataset.get_idx_split() 
train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True)
valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False)
test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False)
evaluator = Evaluator(name="ogbg-molpcba")

def run_exp(config=None):
    wandb.init(job_type="molpcba",
               project="IterativeMethods", 
               config=config, 
               notes="iGCN",
               tags=["iGCN"])
    config = wandb.config
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    wandb.log({
        "device": device_str
    })
    train_schedule = make_uniform_schedule(config.num_iter_layers, config.smooth_fac)
    wandb.log({
        "train_schedule": train_schedule
    })
    model = iterativeGCN_inductive(
            num_tasks=dataset.num_tasks,
            hidden_dim=config.hid_dim,
            train_schedule=train_schedule,
            dropout=0.5)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = OneCycleLR(optimizer, max_lr=config.learning_rate, steps_per_epoch=len(train_loader), epochs=config.num_epochs, pct_start=config.warmup_pct)
    exp_mol(model, optimizer, scheduler,train_loader, valid_loader, test_loader, evaluator, config, device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    wandb.finish()
    
        


sweep_id = wandb.sweep(sweep_config, project="IterativeMethods")
wandb.agent(sweep_id, run_exp, count=100)
    