from __future__ import division
from __future__ import print_function
import numpy as np
import copy
import torch
import torch.nn.functional as F

import sys
from pathlib import Path
BASE_PATH = Path(__file__).parent.parent.absolute()
sys.path.insert(1, str(BASE_PATH))

from src.utils.utils import make_Planetoid_data, make_uniform_schedule, exp_per_model, test
from src.models.iterativeModels import iterativeGCN

import wandb
wandb.login()

'''
This script is not up to date.
'''

sigmoid = F.sigmoid(torch.Tensor(np.linspace(1, 9, 30)))
uniform = np.full(30, 0.97)
linear = np.linspace(0.9, 1, 30)
tanh = F.tanh(torch.Tensor(np.linspace(1, 5, 30)))

def run_exp(hyper=None):
    wandb.init(config=hyper, 
               job_type="schedule_exp", 
               project="IterativeMethods", 
               tags=["iterativeGCN"])
    config = wandb.config
    data, num_features, num_classes = make_Planetoid_data(config, seed=2147483647)
    train_schedule = make_uniform_schedule(config.num_iter_layers, config.smooth_fac)
    model_origin = iterativeGCN(input_dim=num_features,
                                    output_dim=num_classes,
                                    hidden_dim=config.hid_dim,
                                    train_schedule=train_schedule,
                                    dropout=config.dropout,
                                    xavier_init=True
                                    )
    exp_per_model(model_origin, data, config)
    orig_state_dict = copy.deepcopy(model_origin.state_dict())
    
    model_sigm = iterativeGCN(input_dim=num_features,
                            output_dim=num_classes,
                            hidden_dim=config.hid_dim,
                            train_schedule=train_schedule,
                            eval_schedule=sigmoid,
                            dropout=0.5,
                            )
    model_sigm.load_state_dict(orig_state_dict)
    loss_test, acc_test = test(model=model_sigm, data=data)
    wandb.log({
        "sigmoid_acc": acc_test,
        "sigmoid_loss": loss_test
    })

    model_unif = iterativeGCN(input_dim=num_features,
                            output_dim=num_classes,
                            hidden_dim=config.hid_dim,
                            train_schedule=train_schedule,
                            eval_schedule=uniform,
                            dropout=0.5,
                            xavier_init=True
                            )
    model_unif.load_state_dict(orig_state_dict)
    loss_test, acc_test = test(model=model_unif, data=data)
    wandb.log({
        "unif_acc": acc_test,
        "unif_loss": loss_test
    })

    model_line = iterativeGCN(input_dim=num_features,
                            output_dim=num_classes,
                            hidden_dim=config.hid_dim,
                            train_schedule=train_schedule,
                            eval_schedule=linear,
                            dropout=0.5,
                            xavier_init=True
                            )
    model_line.load_state_dict(orig_state_dict)
    loss_test, acc_test = test(model=model_line, data=data)
    wandb.log({
        "line_acc": acc_test,
        "line_loss": loss_test
    })

    model_tanh = iterativeGCN(input_dim=num_features,
                            output_dim=num_classes,
                            hidden_dim=config.hid_dim,
                            train_schedule=train_schedule,
                            eval_schedule=tanh,
                            dropout=0.5,
                            xavier_init=True
                            )
    model_tanh.load_state_dict(orig_state_dict)
    loss_test, acc_test = test(model=model_tanh, data=data)
    wandb.log({
        "tanh_acc": acc_test,
        "tanh_loss": loss_test
    })

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
