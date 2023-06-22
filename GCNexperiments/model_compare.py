from __future__ import division
from __future__ import print_function
from utils import exp_per_model, make_Planetoid_data
from models import iterativeGCN, MLP_GCN, normalNN, iterativeNN, GCN, only_EncDec

import wandb
wandb.login()

def run_exp(config=None):
    data, num_features, num_classes = make_Planetoid_data(config)
    
    wandb.init(config=config, project="IterativeMethods", job_type="model_compare", tags=["iterativeGCN"], reinit=True)
    iterative_gcn = iterativeGCN(input_dim=num_features,
                            output_dim=num_classes,
                            hidden_dim=wandb.config.hid_dim,
                            num_train_iter=wandb.config.num_iter_layers,
                            smooth_fac=wandb.config.smooth_fac,
                            dropout=wandb.config.dropout,
                            xavier_init=True)
    exp_per_model(iterative_gcn, data, wandb.config)
    wandb.finish()
    
    wandb.init(config=config, project="IterativeMethods", job_type="model_compare", tags=["MLP_GCN"], reinit=True)
    mlp_gcn = MLP_GCN(input_dim=num_features,
                  output_dim=num_classes,
                  hidden_dim=wandb.config.hid_dim,
                  num_layers=wandb.config.num_iter_layers,
                  dropout=wandb.config.dropout,
                  xavier_init=True)
    exp_per_model(mlp_gcn, data, wandb.config)
    wandb.finish()

    # wandb.init(config=config, project="IterativeMethods", job_type="model_compare", tags=["NormalNN"], reinit=True)
    # norm_nn = normalNN(input_dim=num_features,
    #               output_dim=num_classes,
    #               hidden_dim=wandb.config.hid_dim,
    #               num_layers=wandb.config.num_iter_layers,
    #               dropout=wandb.config.dropout,
    #               xavier_init=True)
    # exp_per_model(norm_nn, data, wandb.config)
    # wandb.finish()
    
    # wandb.init(config=config, project="IterativeMethods", job_type="model_compare", tags=["iterativeNN"])
    # iter_nn = iterativeNN(input_dim=num_features,
    #               output_dim=num_classes,
    #               hidden_dim=wandb.config.hid_dim,
    #               num_train_iter=wandb.config.num_iter_layers,
    #               smooth_fac=wandb.config.smooth_fac,
    #               dropout=wandb.config.dropout,
    #               xavier_init=True)
    # exp_per_model(iter_nn, data, wandb.config)
    # wandb.finish()

    wandb.init(config=config, project="IterativeMethods", job_type="model_compare", tags=["usualGCN"])
    gcn = GCN(input_dim=num_features,
                output_dim=num_classes,
                hidden_dim=wandb.config.hid_dim,
                num_layers=wandb.config.num_iter_layers,
                dropout=wandb.config.dropout)
    exp_per_model(gcn, data, wandb.config)
    wandb.finish()

    wandb.init(config=config, project="IterativeMethods", job_type="model_compare", tags=["only_EncDec"])
    only_encdec = only_EncDec(input_dim=num_features,
                output_dim=num_classes,
                hidden_dim=wandb.config.hid_dim,
                dropout=wandb.config.dropout,
                xavier_init=True)
    exp_per_model(only_encdec, data, wandb.config)
    wandb.finish()


config = {
    'num_epochs': 200,
    'dataset_name': "Cora",
    'noise_percent': 0.5,
    'hid_dim': 32,
    'num_iter_layers': 10,
    'smooth_fac': 0.7,
    'dropout': 0.5,
    'learning_rate': 0.01,
    'weight_decay': 5e-4
} 

run_exp(config)