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

from src.utils.metrics import MAD
from src.models.iterativeModels import iterativeGCN_inductive2
from src.utils.utils import exp_mol, make_uniform_schedule
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import OneCycleLR, ConstantLR
from torch.optim import Adam

import wandb
wandb.login()


sweep_config = {
    'method': 'grid'
}

metric = {
    'name': 'rocauc',
    'goal': 'maximize'
}
sweep_config['metric'] = metric

parameters_dict = {
    'num_iter_layers': {
        'value': 4
    },
    'learning_rate': {
        'value': 0.0008
    },
    'smooth_fac': {
        'value': 0.8
    },
    'hid_dim': {
        'value': 400
    },
    'weight_decay': {
        'value': 0
    },
    'num_epochs': {
        'value': 100
    },
    'dropout': {
        'value': 0.5
    },
    'dataset_name': {
        'value': 'ogbg-molhiv'
    },
    'warmup_pct': {
        'value': 0.1
    },
    'random_seeds':{
        'values': [12345, 54321, 23456, 65432, 34567, 76543, 45678, 87654, 56789, 98765]
    }
}
sweep_config['parameters'] = parameters_dict

'''
This script is for sweeping for 10 different random seeds.
'''
dataset = PygGraphPropPredDataset(name='ogbg-molhiv') 
split_idx = dataset.get_idx_split() 
train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True)
valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False)
test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False)
evaluator = Evaluator(name="ogbg-molhiv")

def run_exp(config=None):
    with wandb.init(job_type="molhiv",
               project="IterativeMethods", 
               config=config, 
               notes="iGCN",
               tags=["iGCN"]) as run:
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
        train_schedule = make_uniform_schedule(config.num_iter_layers, config.smooth_fac)
        wandb.log({
            "train_schedule": train_schedule
        })
        model = iterativeGCN_inductive2(
                num_tasks=dataset.num_tasks,
                hidden_dim=config.hid_dim,
                train_schedule=train_schedule,
                dropout=config.dropout)
        model = model.to(device)
        optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        #scheduler = OneCycleLR(optimizer, max_lr=config.learning_rate, steps_per_epoch=len(train_loader), epochs=config.num_epochs, pct_start=config.warmup_pct)
        scheduler = ConstantLR(optimizer, factor=1, total_iters=1)
        exp_mol(model, optimizer, scheduler,train_loader, valid_loader, test_loader, evaluator, config, device)
        
        description="IterativeGCN model trained on ogbg-molhiv with random seed " + str(random_seed)
        model_artifact = wandb.Artifact(
                "CSTLR-trained-iGCN-on-molhiv-old", type="model",
                description=description,
                metadata={
                    "num_tasks": dataset.num_tasks,
                    "hidden_dim": config.hid_dim,
                    "train_schedule": train_schedule,
                    "dropout": config.dropout
                })
        model_path = "CSTLR-train_iGCN" + str(random_seed) + ".pth"
        torch.save(model.state_dict(), model_path)
        model_artifact.add_file(model_path)
        wandb.save(model_path)
        run.log_artifact(model_artifact)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    wandb.finish()
    
        


sweep_id = wandb.sweep(sweep_config, project="IterativeMethods")
wandb.agent(sweep_id, run_exp)
    