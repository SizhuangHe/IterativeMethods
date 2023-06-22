from __future__ import division
from __future__ import print_function
import numpy as np
import math
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from utils import accuracy
from models import iterativeGCN, MLP_GCN, normalNN, iterativeNN, GCN

import wandb
wandb.login()


def add_noise(data, percent=0):
    #add random 1's to data
    if percent > 0 and percent <= 1:
        len = np.prod(list(data.x.shape))
        ones = math.floor(len * percent)
        zeros = len - ones
        noise = torch.cat((torch.zeros(zeros), torch.ones(ones)))
        noise = noise[torch.randperm(noise.size(0))]
        noise = torch.reshape(noise, data.x.shape)
        data.x += noise
    return data

def make_data(config):
    dataset = Planetoid(root='data/Planetoid', 
                        name=config['dataset_name'], 
                        transform=NormalizeFeatures())
    data = dataset[0]
    data = add_noise(data, percent=config['noise_percent'])
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    return data, num_features, num_classes


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_epoch(model, data, optimizer):
    model.train()
    output = model(data.x, data.edge_index)
    loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
    pred = output[data.train_mask].argmax(dim=1)
    acc = accuracy(pred, data.y[data.train_mask])
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss, acc

def validate_epoch(model, data):
    model.eval()
    output = model(data.x, data.edge_index)
    loss = F.nll_loss(output[data.val_mask], data.y[data.val_mask])
    pred = output[data.val_mask].argmax(dim=1)
    acc = accuracy(pred, data.y[data.val_mask])
    return loss, acc

def train(model, data, config, num_epochs):
    wandb.watch(model, log="all", log_freq=10)
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    for epoch in range(num_epochs):
        loss_train, acc_train = train_epoch(model, data, optimizer)
        loss_val, acc_val = validate_epoch(model, data)

        wandb.log({
            'training_loss': loss_train,
            'training_accuracy': acc_train,
            'validation_loss': loss_val,
            'validation_accuracy': acc_val,
            "epoch": epoch+1,
        })
  

def test(model, data):
    model.eval()
    output = model(data.x, data.edge_index)
    loss = F.nll_loss(output[data.test_mask], data.y[data.test_mask])
    pred = output[data.test_mask].argmax(dim=1)
    acc = accuracy(pred, data.y[data.test_mask])
    
    
    return loss, acc

def exp_per_model(model, data, config):
    num_params = count_parameters(model)
    wandb.log({ 
            'num_param': num_params
    }) 
    train(model, data, config, config.num_epochs)
    loss_test, acc_test = test(model, data)
    wandb.log({
        'test_loss': loss_test,
        'test_accuracy': acc_test
    })


def run_exp(config=None):
    data, num_features, num_classes = make_data(config)
    
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

    wandb.init(config=config, project="IterativeMethods", job_type="model_compare", tags=["NormalNN"], reinit=True)
    norm_nn = normalNN(input_dim=num_features,
                  output_dim=num_classes,
                  hidden_dim=wandb.config.hid_dim,
                  num_layers=wandb.config.num_iter_layers,
                  dropout=wandb.config.dropout,
                  xavier_init=True)
    exp_per_model(norm_nn, data, wandb.config)
    wandb.finish()
    
    wandb.init(config=config, project="IterativeMethods", job_type="model_compare", tags=["iterativeNN"])
    iter_nn = iterativeNN(input_dim=num_features,
                  output_dim=num_classes,
                  hidden_dim=wandb.config.hid_dim,
                  num_train_iter=wandb.config.num_iter_layers,
                  smooth_fac=wandb.config.smooth_fac,
                  dropout=wandb.config.dropout,
                  xavier_init=True)
    exp_per_model(iter_nn, data, wandb.config)
    wandb.finish()

    wandb.init(config=config, project="IterativeMethods", job_type="model_compare", tags=["usualGCN"])
    gcn = GCN(input_dim=num_features,
                output_dim=num_classes,
                hidden_dim=wandb.config.hid_dim,
                num_layers=wandb.config.num_iter_layers,
                dropout=wandb.config.dropout)
    exp_per_model(gcn, data, wandb.config)
    wandb.finish()


config = {
    'num_epochs': 200,
    'dataset_name': "Cora",
    'noise_percent': 0.5,
    'hid_dim': 32,
    'num_iter_layers': 7,
    'smooth_fac': 0.7,
    'dropout': 0.5,
    'learning_rate': 0.01,
    'weight_decay': 5e-4
} 

run_exp(config)