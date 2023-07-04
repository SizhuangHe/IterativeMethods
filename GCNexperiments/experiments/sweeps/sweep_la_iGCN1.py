from __future__ import division
from __future__ import print_function
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

import sys
from pathlib import Path
BASE_PATH = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(1, str(BASE_PATH))

from src.utils.utils import make_Planetoid_data, exp_per_model, make_uniform_schedule
from src.models.iterativeModels import learnable_adaptive_iGCN

import wandb
wandb.login()

'''
This script is to sweep for the best set of hyperparameters for iterativeGCN,
given Cora dataset with a fixed amount of noise.
'''

def run_exp(config=None):
    wandb.init(job_type="Sweep", 
               project="IterativeMethods", 
               config=config, 
               notes="la_iGCN",
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
        'value': 0.004
    },
    'hid_dim': {
        'value': 32
    },
    'weight_decay': {
        'value': 4e-4
    },
    'num_epochs': {
        'value': 200
    },
    'dropout': {
        'value': 0.5
    },
    'dataset_name': {
        'value': 'Cora'
    },
    'noise_percent': {
        'value': 0.1
    }
}
sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project="IterativeMethods")
wandb.agent(sweep_id, run_exp, count=400)
    
        