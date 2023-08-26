from __future__ import division
from __future__ import print_function
import numpy as np
import torch

import sys
from pathlib import Path
BASE_PATH = Path(__file__).parent.parent.parent.parent.absolute()
sys.path.insert(1, str(BASE_PATH))
import torch
from src.utils.utils import make_Planetoid_data, exp_per_model, make_uniform_schedule
from src.models.iterativeModels import iterativeGAT
from src.utils.metrics import MAD
from argparse import ArgumentParser

import wandb
wandb.login()

parser = ArgumentParser()
parser.add_argument("--hid_dim", type=int, help="the hidden dimension of the model", default=32)
parser.add_argument("--num_epochs", type=int, help="number of training epochs", default=200)
parser.add_argument("--dataset", type=str, help="name of the Planetoid dataset, choose from Cora, CiteSeer and PubMed", 
                    choices=["Cora", "CiteSeer", "PubMed"], 
                    default="Cora")
parser.add_argument("--noise", type=float, help="the amount of noise to add to the dataset", choices=[0, 0.5, 0.7], default=0)
args = parser.parse_args()

'''
This script is for sweeping for a set of hyperparameters for the iterative GAT,
on the Cora dataset with a fixed amount of noise.
'''

def run_exp(config=None):
    wandb.init(job_type="Sweep", 
               project="IterativeMethods", 
               config=config, 
               notes="Sweep for the iGAT",
               tags=["iterativeGAT"])
    config = wandb.config
    data, num_features, num_classes = make_Planetoid_data(config, seed=2147483647)
    train_schedule = make_uniform_schedule(config.num_iter_layers, config.smooth_fac)
    model =iterativeGAT(input_dim=num_features,
                output_dim=num_classes,
                hidden_dim=config.hid_dim,
                train_schedule=train_schedule,
                heads=8,
                dropout=config.dropout,
                attn_dropout_rate=config.dropout,
                xavier_init=True
    )
    exp_per_model(model, data, config)

    out = model(data.x, data.edge_index)
    mad = MAD(out.detach())
    wandb.log({
        "MAD": mad
    })

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
        'value': 2
    },
    'learning_rate': {
        'value': 0.0145
    },
    'smooth_fac': {
        'value': 0.55
    },
    'hid_dim': {
        'value': args.hid_dim
    },
    'weight_decay': {
        'value': 5e-4
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
    'noise_percent': {
        'value': args.noise
    },
    'noise_seed':{
        'value': 2147483647
    }
}
sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project="IterativeMethods")
wandb.agent(sweep_id, run_exp, count=500)
    
        