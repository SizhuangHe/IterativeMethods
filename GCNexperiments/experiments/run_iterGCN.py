from __future__ import division
from __future__ import print_function
import numpy as np
import torch
import torch.nn.functional as F

import sys
from pathlib import Path
BASE_PATH = Path(__file__).parent.parent.absolute()
sys.path.insert(1, str(BASE_PATH))

from src.utils.utils import make_Planetoid_data, train, test, make_uniform_schedule
from src.models.GCN import GCN
from src.models.iterativeModels import iterativeGCN
import wandb
wandb.login()

'''
This script is to experiment on the performance of iterative GCN with given hyperparameters, after sweep.
'''

def run_exp(hyper=None, train_schedule=None):
    wandb.init(config=hyper, job_type="iter_layer_vs_perfm", project="IterativeMethods", tags=["iterativeGCN"])
    config = wandb.config
    data, num_features, num_classes = make_Planetoid_data(config)
    wandb.log({
        'train_schedule': train_schedule
    })
    
    iterative_gcn = iterativeGCN(input_dim=num_features,
                                    output_dim=num_classes,
                                    hidden_dim=config.hid_dim,
                                    train_schedule=train_schedule,
                                    dropout=config.dropout)
    loss_test, acc_test = test(iterative_gcn, data)
    train(iterative_gcn, data, config)
    wandb.log({
            'test_loss': loss_test,
            'test_accuracy': acc_test,
            'iteration': iter
        })
    wandb.finish()

        
config = {
    'num_epochs': 200,
    'dataset_name': "Cora",
    'noise_percent': 0.4,
    'hid_dim': 32,
    'num_iter_layers': 4,
    'smooth_fac': 0.45,
    'dropout': 0.5,
    'learning_rate': 0.002,
    'weight_decay': 4e-4
} 

schedule = F.sigmoid(torch.Tensor(np.linspace(0.5, 3, 7))).detach().cpu().numpy()
run_exp(config, schedule)