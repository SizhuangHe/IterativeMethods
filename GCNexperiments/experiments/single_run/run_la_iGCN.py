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
from src.models.iterativeModels import learnable_adaptive_iGCN
import wandb
wandb.login()

'''
This script is to experiment on the performance of iterative GCN with a given set of hyperparameters.
'''

def run_exp(hyper=None):
    wandb.init(config=hyper, 
               job_type="run_iGCN", 
               project="IterativeMethods", 
               tags=["la_iGCN"])
    config = wandb.config
    data, num_features, num_classes = make_Planetoid_data(config)
    
    model = learnable_adaptive_iGCN(input_dim=num_features,
                                    output_dim=num_classes,
                                    hidden_dim=config.hid_dim,
                                    num_iterations=config.num_iter_layers,
                                    dropout=config.dropout)
    exp_per_model(model, data, config)
    smooth_fac = model.train_schedule.detach().numpy().copy()
    wandb.log({
        'learned smoothing factors': smooth_fac
    })
    print('learned smoothing factors: ', smooth_fac)
    
    wandb.finish()

        
config = {
    'num_epochs': 500,
    'dataset_name': "Cora",
    'noise_percent': 0.4,
    'hid_dim': 32,
    'num_iter_layers': 20,
    'dropout': 0.5,
    'learning_rate': 0.005,
    'weight_decay': 4e-4
} 


run_exp(config)