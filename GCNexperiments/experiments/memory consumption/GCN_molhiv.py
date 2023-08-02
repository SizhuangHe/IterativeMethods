from __future__ import division
from __future__ import print_function
import numpy as np
import torch

import sys
from pathlib import Path
BASE_PATH = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(1, str(BASE_PATH))
import torch


from src.models.models import GCN_inductive
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.loader import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm

import wandb
wandb.login()

config = {
    'num_iter_layers': 5,
    'learning_rate': 0.001,
    'smooth_fac': 0.5,
    'hid_dim': 300,
    'weight_decay': 0,
    'num_epochs': 100,
    'dropout': 0.6,
    'dataset_name': 'ogbg-molhiv',

}

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
    
    with profile(activities=[ProfilerActivity.CPU],
        profile_memory=True, record_shapes=True) as prof:
        for step, batched_data in enumerate(tqdm(train_loader)):
            batched_data = batched_data.to(device)
            with torch.no_grad():
                pred = model(batched_data.x, batched_data.edge_index, batched_data.batch)
            
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
    wandb.finish()
    
        


run_exp(config)
        