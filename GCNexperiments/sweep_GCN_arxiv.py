from __future__ import division
from __future__ import print_function
import numpy as np
import torch

import sys
from pathlib import Path
BASE_PATH = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(1, str(BASE_PATH))
from src.utils.run_exp import run_arxiv

import wandb
wandb.login()

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
        'value': 3
    },
    'learning_rate': {
        'value': 0.01
    },
    'smooth_fac': {
        'values': [0.3, 0.5, 0.7, 0.75, 0.8]
    },
    'hid_dim': {
        'value': 256
    },
    'weight_decay': {
        'values': [0, 0.0001]
    },
    'num_epochs': {
        'value': 500
    },
    'dropout': {
        'value': 0.5
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
wandb.agent(sweep_id, run_arxiv, count=1)