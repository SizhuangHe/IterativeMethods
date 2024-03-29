from __future__ import division
from __future__ import print_function
import numpy as np
import torch

import sys
from pathlib import Path
BASE_PATH = Path(__file__).parent.parent.parent.parent.parent.absolute()
sys.path.insert(1, str(BASE_PATH))
import torch
from src.utils.utils import exp_vocsp, make_uniform_schedule
from src.models.models import GCN_vocsp
from src.utils.metrics import MAD
from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from argparse import ArgumentParser

import wandb
wandb.login()

parser = ArgumentParser()
parser.add_argument("--hid_dim", type=int, help="the hidden dimension of the model", default=220)
args = parser.parse_args()

train_dataset = LRGBDataset(root="/vast/palmer/scratch/dijk/sh2748/data/palmer_scratch/data/LRGB", name="PascalVOC-SP", split="train")
val_dataset = LRGBDataset(root="/vast/palmer/scratch/dijk/sh2748/data/palmer_scratch/data/LRGB", name="PascalVOC-SP", split="val")
test_dataset = LRGBDataset(root="/vast/palmer/scratch/dijk/sh2748/data/palmer_scratch/data/LRGB", name="PascalVOC-SP", split="test")
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

def run_exp(config=None):
    wandb.init(job_type="Sweep", 
               project="IterativeMethods", 
               config=config, 
               notes="Sweep for the usual GCN, from McCleary",
               tags=["usualGCN"])
    config = wandb.config
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    wandb.log({
        "device": device_str
    })
    

    model = GCN_vocsp(out_dim=train_dataset.num_classes,
                          hidden_dim=config.hid_dim,
                          MLP_layers=3,
                          num_layers=config.num_iter_layers,
                          dropout=config.dropout
                          ).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=10, min_lr=config.min_lr, verbose=True)
    exp_vocsp(model, optimizer, scheduler, train_loader, val_loader, test_loader, config.num_epochs, device)

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
        'value': 4
    },
    'learning_rate': {
        'values': [0.0001, 0.0005, 0.001, 0.0015, 0.002]
    },
    'smooth_fac': {
        'value': 0.5 #doesn't matter
    },
    'hid_dim': {
        'value': args.hid_dim
    },
    'weight_decay': {
        'values': [0, 1e-5]
    },
    'num_epochs': {
        'value': 200
    },
    'dropout': {
        'value': 0.0
    },
    'dataset_name': {
        'value': 'VOC-SP'
    },
    'min_lr':{
        'value': 1e-5
    }
}
sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project="IterativeMethods")
wandb.agent(sweep_id, run_exp)
    
        