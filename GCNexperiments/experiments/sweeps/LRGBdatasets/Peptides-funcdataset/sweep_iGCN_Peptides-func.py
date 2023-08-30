from __future__ import division
from __future__ import print_function
import numpy as np
import torch

import sys
from pathlib import Path
BASE_PATH = Path(__file__).parent.parent.parent.parent.parent.absolute()
sys.path.insert(1, str(BASE_PATH))
import torch
from src.utils.utils import exp_pepfunc, make_uniform_schedule
from src.models.iterativeModels import iterativeGCN_peptides
from src.utils.metrics import MAD
from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from argparse import ArgumentParser

import wandb
wandb.login()

parser = ArgumentParser()
parser.add_argument("--hid_dim", type=int, help="the hidden dimension of the model", default=220)
parser.add_argument("--dataset", type=str, help="the name of the LRGB dataset, choose from pascal, coco and peptides", 
                    choices=['PascalVOC-SP', 'COCO-SP', 'Peptides-func'], 
                    default='PascalVOC-SP')
parser.add_argument("--lr_sche", type=str, help="type of learning rate scheduler, choose from reduce-on-plateau and one-cycle", 
                    choices=["reduce-on-lateau", "one-cycle"], 
                    default="reduce-on-plateau")
args = parser.parse_args()

train_dataset = LRGBDataset(root="/vast/palmer/scratch/dijk/sh2748/data/LRGB", name=args.dataset, split="train")
val_dataset = LRGBDataset(root="/vast/palmer/scratch/dijk/sh2748/data/LRGB", name=args.dataset, split="val")
test_dataset = LRGBDataset(root="/vast/palmer/scratch/dijk/sh2748/data/LRGB", name=args.dataset, split="test")
train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)



def run_exp(config=None):
    wandb.init(job_type="Sweep", 
               project="IterativeMethods", 
               config=config, 
               notes="Sweep for the iterative GCN, from McCleary",
               tags=["iGCN"],
               dir="/vast/palmer/scratch/dijk/sh2748")
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

    model = iterativeGCN_peptides(out_dim=train_dataset.num_classes,
                          hidden_dim=config.hid_dim,
                          train_schedule=train_schedule,
                          MLP_layers=3,
                          dropout=config.dropout
                          ).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    if args.lr_sche == "reduce-on-lateau":
        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=10, min_lr=config.learning_rate/50, verbose=True)
    elif args.lr_sche == "one-cycle":
        scheduler = OneCycleLR(optimizer, max_lr=config.learning_rate, epochs=config.num_epochs, steps_per_epoch=len(train_loader))
    else:
        raise Exception("Invalid learning rate scheduler.")
    exp_pepfunc(model, optimizer, scheduler, train_loader, val_loader, test_loader, config.num_epochs, device=device)

    wandb.finish()
    

sweep_config = {
    'method': 'grid'
}

metric = {
    'name': 'AP',
    'goal': 'maximize'
}
sweep_config['metric'] = metric

parameters_dict = {
    'num_iter_layers': {
        'values': [8, 14, 20, 27]
    },
    'learning_rate': {
        'values': [0.0001, 0.0005, 0.001]
    },
    'smooth_fac': {
        'values': [0.5, 0.7, 0.8] 
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
        'value': 0.5
    },
    'dataset_name': {
        'value': args.dataset
    }
}
sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project="IterativeMethods")
wandb.agent(sweep_id, run_exp)
    
        