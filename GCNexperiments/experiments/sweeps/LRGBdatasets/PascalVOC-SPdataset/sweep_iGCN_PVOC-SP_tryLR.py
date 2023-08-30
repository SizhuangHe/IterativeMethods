from __future__ import division
from __future__ import print_function
import numpy as np
import torch

import sys
from pathlib import Path
BASE_PATH = Path(__file__).parent.parent.parent.parent.parent.absolute()
sys.path.insert(1, str(BASE_PATH))
import torch
from src.utils.utils import make_uniform_schedule, weighted_cross_entropy, count_parameters
from src.models.iterativeModels import iterativeGCN_vocsp
from src.utils.metrics import MAD
from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from argparse import ArgumentParser
from sklearn.metrics import f1_score

import wandb
wandb.login()

train_dataset = LRGBDataset(root="/vast/palmer/scratch/dijk/sh2748/data/palmer_scratch/data/LRGB", name="PascalVOC-SP", split="train")
val_dataset = LRGBDataset(root="/vast/palmer/scratch/dijk/sh2748/data/palmer_scratch/data/LRGB", name="PascalVOC-SP", split="val")
test_dataset = LRGBDataset(root="/vast/palmer/scratch/dijk/sh2748/data/palmer_scratch/data/LRGB", name="PascalVOC-SP", split="test")
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

parser = ArgumentParser()
parser.add_argument("--hid_dim", type=int, help="the hidden dimension of the model", default=220)
args = parser.parse_args()

def train_vocsp_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    criterion = weighted_cross_entropy
    epoch_loss = 0
    for step, batched_data in enumerate(loader):  # Iterate in batches over the training dataset.
        batched_data = batched_data.to(device)
        pred = model(batched_data.x, batched_data.edge_index, batched_data.edge_attr,batched_data.batch) # size of pred is [number of nodes, number of features]
        true = batched_data.y
        loss = criterion(pred, true)
        epoch_loss += loss.item()
        optimizer.zero_grad()  
        loss.backward() 
        optimizer.step()
        scheduler.step()
        
    return epoch_loss

def eval_vocsp(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    criterion = weighted_cross_entropy
    val_loss = 0
    for step, batched_data in enumerate(loader):  # Iterate in batches over the training dataset.
        batched_data = batched_data.to(device)
        pred = model(batched_data.x, batched_data.edge_index, batched_data.edge_attr,batched_data.batch) # size of pred is [number of nodes, number of features]
        true = batched_data.y
        loss = criterion(pred, true)
        val_loss += loss.item()

        pred_val = pred.max(dim=1)[1] # pred_val contains actually class predictions
        y_pred.append(pred_val.detach())
        y_true.append(true.detach())
    
    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_pred = torch.cat(y_pred, dim = 0).cpu().numpy()
    val_f1 = f1_score(y_true, y_pred, average="macro")
        
    return val_loss, val_f1

def train_vocsp(model, optimizer, scheduler, train_loader, valid_loader, num_epochs, device):
    wandb.watch(model, log="all", log_freq=10)
    for epoch in range(num_epochs):
        train_loss = train_vocsp_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_f1 = eval_vocsp(model, valid_loader, device)
        
        wandb.log({
            "Train loss": train_loss,
            "Validate f1": val_f1,
            "Validate loss": val_loss,
            "epoch": epoch+1,
            "lr": optimizer.param_groups[0]['lr']
        })

def exp_vocsp(model, optimizer, scheduler,train_loader, valid_loader, test_loader, num_epochs,device):
    num_params = count_parameters(model)
    wandb.log({ 
            'num_param': num_params
    }) 
    train_vocsp(model, optimizer, scheduler,train_loader, valid_loader, num_epochs, device)
    test_loss, test_f1=eval_vocsp(model, test_loader, device)
    wandb.log({
        "Test loss": test_loss,
        "Test f1": test_f1
    })
    



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

    model = iterativeGCN_vocsp(out_dim=train_dataset.num_classes,
                          hidden_dim=config.hid_dim,
                          train_schedule=train_schedule,
                          MLP_layers=3,
                          dropout=config.dropout,
                          xavier_init=True
                          ).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = OneCycleLR(optimizer, max_lr=config.learning_rate, epochs=config.num_epochs, steps_per_epoch=len(train_loader), pct_start=config.pct_start)
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
        'values': 8
    },
    'learning_rate': {
        'values': [0.00001, 0.00005, 0.0001]
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
        'value': 'VOC-SP'
    },
    'pct_start':{
        'value': 0.2
    }
}
sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project="IterativeMethods")
wandb.agent(sweep_id, run_exp)
    
        