from __future__ import division
from __future__ import print_function
import numpy as np
import torch
import torch.nn.functional as F

import sys
from pathlib import Path
BASE_PATH = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(1, str(BASE_PATH))

from src.utils.utils import make_Planetoid_data, exp_per_model, make_uniform_schedule
from src.models.iterativeModels import iterativeGCN
import wandb
wandb.login()

'''
This script is to experiment on the performance of iterative GCN with a given set of hyperparameters.
'''

def run_exp(hyper=None):
    wandb.init(config=hyper, 
               job_type="run_iGCN", 
               project="IterativeMethods", 
               tags=["iterativeGCN"])
    config = wandb.config
    train_schedule = make_uniform_schedule(config.num_iter_layers, config.smooth_fac)
    data, num_features, num_classes = make_Planetoid_data(config)
    wandb.log({
        'train_schedule': train_schedule
    })
    
    model = iterativeGCN(input_dim=num_features,
                                    output_dim=num_classes,
                                    hidden_dim=config.hid_dim,
                                    train_schedule=train_schedule,
                                    dropout=config.dropout)
    exp_per_model(model, data, config)
    wandb.finish()

        
config = {
    'num_epochs': 200,
    'dataset_name': "Cora",
    'noise_percent': 0.5,
    'hid_dim': 32,
    'num_iter_layers': 4,
    'smooth_fac': 0.6,
    'dropout': 0.5,
    'learning_rate': 0.005,
    'weight_decay': 4e-4
} 


run_exp(config)