from __future__ import division
from __future__ import print_function
import numpy as np
import torch
import torch.nn.functional as F
from utils import make_Planetoid_data, train, test, make_uniform_schedule
from models import iterativeGCN, GCN
import wandb
wandb.login()

'''
In this script, we train and evaluate iterativeGCNs and usual GCNs with different iterations/layers
and see how performance change over iterations/layers
'''

def run_exp(hyper=None):
    wandb.init(config=hyper, job_type="iter_layer_vs_perfm", project="IterativeMethods", tags=["iterativeGCN"])
    config = wandb.config
    data, num_features, num_classes = make_Planetoid_data(config)
    
    
    for iter in range(1, 11):
        train_schedule = make_uniform_schedule(iter, config.smooth_fac)
        iterative_gcn = iterativeGCN(input_dim=num_features,
                                    output_dim=num_classes,
                                    hidden_dim=config.hid_dim,
                                    train_schedule=train_schedule,
                                    dropout=config.dropout)
        train(iterative_gcn, data, config)
        loss_test, acc_test = test(iterative_gcn, data)
        wandb.log({
            'test_loss': loss_test,
            'test_accuracy': acc_test,
            'iteration': iter
        })
        del iterative_gcn
    wandb.finish()
    
    wandb.init(config=hyper, job_type="iter_layer_vs_perfm", project="IterativeMethods", tags=["usualGCN"])
    config = wandb.config
    for layer in range(1,11):
        gcn = GCN(input_dim=num_features,
                output_dim=num_classes,
                hidden_dim=config.hid_dim,
                num_layers=layer,
                dropout=config.dropout)
        train(gcn, data, config)
        loss_test, acc_test = test(gcn, data)
        wandb.log({
            'test_loss': loss_test,
            'test_accuracy': acc_test,
            'layer': layer
        })
        del gcn
    wandb.finish()

        
config = {
    'num_epochs': 200,
    'dataset_name': "Cora",
    'noise_percent': 0.4,
    'hid_dim': 32,
    'num_iter_layers': 6,
    'smooth_fac': 0.525,
    'dropout': 0.5,
    'learning_rate': 0.011,
    'weight_decay': 4e-4
} 

run_exp(config)