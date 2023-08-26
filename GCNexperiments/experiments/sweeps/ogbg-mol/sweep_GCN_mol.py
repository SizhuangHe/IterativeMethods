from __future__ import division
from __future__ import print_function
import numpy as np
import torch
from random import seed

import sys
from pathlib import Path
BASE_PATH = Path(__file__).parent.parent.parent.parent.absolute()
sys.path.insert(1, str(BASE_PATH))
import torch

from src.utils.metrics import MAD
from src.models.models import GCN_inductive
from src.utils.utils import exp_mol
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import OneCycleLR, ConstantLR
from torch.optim import Adam
from argparse import ArgumentParser

import wandb
wandb.login()

parser = ArgumentParser()
parser.add_argument("--hid_dim", type=int, help="the hidden dimension of the model", default=200)
parser.add_argument("--num_epochs", type=int, help="number of training epochs", default=100)
parser.add_argument("--dataset", type=str, help="name of the OGB-mol dataset, choose from ogbg-molhiv and ogbg-molpcba", 
                    choices=["ogbg-molhiv", "ogbg-molpcba"], 
                    default="ogbg-molhiv")
parser.add_argument("--lr_sche", type=str, help="type of learning rate scheduler, choose from one-cycle and constant", 
                    choices=["constant", "one-cycle"], 
                    default="constant")
args = parser.parse_args()


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
        'value': args.hid_dim
    },
    'weight_decay': {
        'value': 0
    },
    'num_epochs': {
        'value': args.num_epochs
    },
    'dropout': {
        'value': 0.6
    },
    'dataset_name': {
        'value': args.dataset
    },
    'random_seeds':{
        'values': [123, 234, 345, 456, 567, 678, 789, 879, 901, 12]
    },
    "warmup_pct":{
        'value': 0.15
    },
    'scheduler':{
        'value': "Constant"
    }
}
sweep_config['parameters'] = parameters_dict

'''
This script is for sweeping for a set of hyperparameters for the usual GCN,
on the Cora dataset with a fixed amount of noise.
'''
dataset = PygGraphPropPredDataset(name=args.dataset) 
split_idx = dataset.get_idx_split() 
train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True)
valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False)
test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False)
evaluator = Evaluator(name=args.dataset)

def run_exp(config=None):
    with wandb.init(job_type="molhiv",
               project="IterativeMethods", 
               config=config, 
               notes="usualGCN",
               tags=["usualGCN"]) as run:
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
                dropout=config.dropout)
        model = model.to(device)
        optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        if args.lr_sche == "constant": 
            scheduler = ConstantLR(optimizer, factor=1, total_iters=1)
        elif args.lr_sche == "one-cycle":
            scheduler = OneCycleLR(optimizer, max_lr=config.learning_rate, steps_per_epoch=len(train_loader), epochs=config.num_epochs, pct_start=config.warmup_pct)
        else:
            raise Exception("Invalid learning rate scheduler.")    
        exp_mol(model, optimizer, scheduler,train_loader, valid_loader, test_loader, evaluator, config, device)
        
        description="GCN model trained on ogbg-molhiv with random seed " + str(random_seed)
        model_artifact = wandb.Artifact(
                "trained-GCN-on-molhiv-200-CSTLR", type="model",
                description=description,
                metadata={
                    "num_tasks": dataset.num_tasks,
                    "hidden_dim": config.hid_dim,
                    "num_layers":config.num_iter_layers,
                    "dropout": config.dropout
                })
        model_path = "train_GCN-200-CSTLR" + str(random_seed) + ".pth"
        torch.save(model.state_dict(), model_path)
        model_artifact.add_file(model_path)
        wandb.save(model_path)
        run.log_artifact(model_artifact)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    wandb.finish()
    
        


sweep_id = wandb.sweep(sweep_config, project="IterativeMethods")
wandb.agent(sweep_id, run_exp)
    
        