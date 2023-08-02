from __future__ import division
from __future__ import print_function
import numpy as np
import torch
import yaml

import sys
from pathlib import Path
BASE_PATH = Path(__file__).parent.parent.parent.parent.absolute()
sys.path.insert(1, str(BASE_PATH))

from src.utils.utils import make_Planetoid_data, exp_per_model
from src.models.models import GCN
from src.utils.metrics import MAD


import wandb
wandb.login()

'''
This script is for sweeping for a set of hyperparameters for the usual GCN,
on the Cora dataset with a fixed amount of noise.
'''

def run_exp(config=None):
    with open('./sweep_test.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    wandb.init(job_type="SweepTest", 
               project="IterativeMethods", 
               config=config, 
               notes="test CLI",
               tags=["usualGCN"])
    config = wandb.config
    data, num_features, num_classes = make_Planetoid_data(config, seed=2147483647)
    model = GCN(
        input_dim=num_features,
        output_dim=num_classes,
        hidden_dim=config.hid_dim,
        num_layers=config.num_iter_layers,
        dropout=config.dropout
    )
    exp_per_model(model, data, config)

    out = model(data.x, data.edge_index)
    mad = MAD(out.detach())
    wandb.log({
        "MAD": mad
    })

    wandb.finish()
    
if __name__ == "__main__":   
    run_exp()