from __future__ import division
from __future__ import print_function


import sys
from pathlib import Path
BASE_PATH = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(1, str(BASE_PATH))

from src.utils.utils import make_Planetoid_data, exp_per_model
from src.models.GCN import GCN

import wandb
wandb.login()

'''
This script is to experiment on the performance of GCN variant with a given set of hyperparameters.
'''

def run_exp(config=None):
    wandb.init(job_type="run_iGCNv", 
               project="IterativeMethods", 
               config=config, 
               notes="variant of iGCN experiments",
               tags=["iterativeGCNvariant"])
    config = wandb.config
    data, num_features, num_classes = make_Planetoid_data(config)
    model = GCN(input_dim=num_features,
                                 output_dim=num_classes,
                                 hidden_dim=config.hid_dim,
                                 num_layers=config.num_iter_layers,
                                 dropout=config.dropout,
                                 )
    exp_per_model(model, data, config)
    wandb.finish()
    

config = {
    'num_epochs': 200,
    'dataset_name': "Cora",
    'noise_percent': 0.7,
    'hid_dim': 32,
    'num_iter_layers': 2,
    'smooth_fac': 0.7,
    'dropout': 0.5,
    'learning_rate': 0.01,
    'weight_decay': 4e-4
} 
run_exp(config)
        