from __future__ import division
from __future__ import print_function
import numpy as np
import torch
from random import seed
import sys
from pathlib import Path
BASE_PATH = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(1, str(BASE_PATH))
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ConstantLR

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
dataset = PygGraphPropPredDataset(name="ogbg-molpcba") 
split_idx = dataset.get_idx_split() 
train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True)
valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False)
test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False)
evaluator = Evaluator(name="ogbg-molpcba")

def run_exp(config=None):
    wandb.init(job_type="molpcba",
               project="IterativeMethods", 
               config=config, 
               notes="usualGCN",
               tags=["usualGCN"])
    config = wandb.config
    random_seed = config.random_seeds
    torch.manual_seed(random_seed)
    seed(random_seed)
    np.random.seed(random_seed)
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
    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = ConstantLR(optimizer, factor=1, total_iters=1)
    exp_mol(model, optimizer, scheduler, train_loader, valid_loader, test_loader, evaluator, config, device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    wandb.finish()
    
        

sweep_config = {
    'method': 'grid'
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
        'value': 'ogbg-molpcba'
    },
    'noise_percent': {
        'value': 0
    },
    'random_seeds':{
        'values': [12345, 54321, 23456, 65432, 34567, 76543, 45678, 87654, 56789, 98765]
    }
}
sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project="IterativeMethods")
wandb.agent(sweep_id, run_exp)
    
        