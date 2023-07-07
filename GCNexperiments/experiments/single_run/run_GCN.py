from __future__ import division
from __future__ import print_function


import sys
from pathlib import Path
BASE_PATH = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(1, str(BASE_PATH))
import torch
from src.utils.utils import make_Planetoid_data, exp_per_model
from src.models.GCN import GCN
from src.utils.metrics import MAD

import wandb
wandb.login()

'''
This script is to experiment on the performance of GCN variant with a given set of hyperparameters.
'''

def run_exp(config=None):
    wandb.init(job_type="run_GCN", 
               project="IterativeMethods", 
               config=config, 
               notes="usualGCN",
               tags=["usualGCN"])
    config = wandb.config
    data, num_features, num_classes = make_Planetoid_data(config)
    model = GCN(input_dim=num_features,
                                 output_dim=num_classes,
                                 hidden_dim=config.hid_dim,
                                 num_layers=config.num_iter_layers,
                                 dropout=config.dropout,
                                 )
    exp_per_model(model, data, config)

    out = model(data.x, data.edge_index)
    torch.set_printoptions(profile="full")
    print(out)
    mad = MAD(out.detach())
    wandb.log({
        "MAD": mad
    })
    print("MAD: ", mad)

    wandb.finish()
    

config = {
    'num_epochs': 200,
    'dataset_name': "Cora",
    'noise_percent': 0,
    'hid_dim': 32,
    'num_iter_layers': 9,
    'smooth_fac': 0.7,
    'dropout': 0.5,
    'learning_rate': 0.01,
    'weight_decay': 4e-4
} 
run_exp(config)
        