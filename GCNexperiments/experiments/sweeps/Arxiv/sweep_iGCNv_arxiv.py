from __future__ import division
from __future__ import print_function
import numpy as np
import torch

import sys
from pathlib import Path
BASE_PATH = Path(__file__).parent.parent.parent.parent.absolute()
sys.path.insert(1, str(BASE_PATH))
from src.utils.run_exp import run_arxiv_iGCN
from argparse import ArgumentParser

import wandb
wandb.login()

parser = ArgumentParser()
parser.add_argument("--hid_dim", type=int, help="the hidden dimension of the model", default=256)
parser.add_argument("--num_epochs", type=int, help="number of training epochs", default=500)
args = parser.parse_args()

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
        'values': [3,4,5,6,7,8,9]
    },
    'learning_rate': {
        'values': [0.001, 0.003, 0.005, 0.007, 0.009, 0.01]
    },
    'smooth_fac': {
        'values': [0.3, 0.5, 0.7, 0.75, 0.8]
    },
    'hid_dim': {
        'value': args.hid_dim
    },
    'weight_decay': {
        'values': [0, 0.0001, 0.001]
    },
    'num_epochs': {
        'value': args.num_epochs
    },
    'dropout': {
        'values': [0.3, 0.5]
    },
    'dataset_name': {
        'value': 'ogbn-arxiv'
    },
    'warmup_pct': {
        'values': [0.05, 0.1, 0.15, 0.2]
    }
}
sweep_config['parameters'] = parameters_dict



sweep_id = wandb.sweep(sweep_config, project="IterativeMethods")
wandb.agent(sweep_id, run_arxiv_iGCN, count=100)