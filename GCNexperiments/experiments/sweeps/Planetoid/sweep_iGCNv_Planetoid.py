from __future__ import division
from __future__ import print_function
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

import sys
from pathlib import Path
BASE_PATH = Path(__file__).parent.parent.parent.parent.sabsolute()
sys.path.insert(1, str(BASE_PATH))

from src.utils.utils import build_iterativeGCN, make_Planetoid_data, exp_per_model, make_uniform_schedule
from src.models.variantModels import iterativeGCN_variant
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
This script is to sweep for the best set of hyperparameters for iterativeGCN,
given Cora dataset with a fixed amount of noise.
'''

def run_exp(config=None):
    wandb.init(job_type="Sweep", 
               project="IterativeMethods", 
               config=config, 
               notes="variant of iGCN experiments, from greatlakes",
               tags=["iterativeGCNvariant"])
    config = wandb.config
    train_schedule = make_uniform_schedule(config.num_iter_layers, config.smooth_fac)
    wandb.log({
        "train_schedule": train_schedule
    })
    data, num_features, num_classes = make_Planetoid_data(config, seed=2147483647)
    model = iterativeGCN_variant(input_dim=num_features,
                                 output_dim=num_classes,
                                 hidden_dim=config.hid_dim,
                                 train_schedule=train_schedule,
                                 dropout=config.dropout,
                                 xavier_init=True
                                 )
    exp_per_model(model, data, config)

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
        'values': [2, 3, 4, 5, 6, 7, 8, 9]
    },
    'learning_rate': {
        'value': 0.0115
    },
    'smooth_fac': {
        'value': 0.9
    },
    'hid_dim': {
        'value': args.hid_dim
    },
    'weight_decay': {
        'value': 3e-4
    },
    'num_epochs': {
        'value': args.num_epochs
    },
    'dropout': {
        'value': 0.5
    },
    'dataset_name': {
        'value': args.dataset
    },
    'noise_percent': {
        'value': args.noise
    }
}
sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project="IterativeMethods")
wandb.agent(sweep_id, run_exp, count=200)
    
        