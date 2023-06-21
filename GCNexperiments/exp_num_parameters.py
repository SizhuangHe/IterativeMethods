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
from models import iterativeGCN, GCN

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
                        name=config.dataset_name, 
                        transform=NormalizeFeatures())
    data = dataset[0]
    data = add_noise(data, percent=config.noise_percent)
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    return data, num_features, num_classes

def make_models(config, input_dim, output_dim):
    iterative_model = iterativeGCN(input_dim=input_dim,
                            output_dim=output_dim,
                            hidden_dim=config.hid_dim,
                            num_train_iter=config.num_iterations,
                            smooth_fac=config.smooth_fac,
                            dropout=config.dropout)
    normal_model = GCN(input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=config.hid_dim,
                num_layers=config.num_layers,
                dropout=config.dropout)
    return iterative_model, normal_model



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

def train(model, data, config, num_epochs, iter=True):
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    for epoch in range(num_epochs):
        loss_train, acc_train = train_epoch(model, data, optimizer)
        loss_val, acc_val = validate_epoch(model, data)

    if iter:
        wandb.log({
            'i_training_loss': loss_train,
            'i_training_accuracy': acc_train,
            'i_validation_loss': loss_val,
            'i_validation_accuracy': acc_val,
            "epoch": epoch+1,
        })
    else:
        wandb.log({
            'n_training_loss': loss_train,
            'n_training_accuracy': acc_train,
            'n_validation_loss': loss_val,
            'n_validation_accuracy': acc_val,
            "epoch": epoch+1,
        })
  

def test(model, data):
    model.eval()
    output = model(data.x, data.edge_index)
    loss = F.nll_loss(output[data.test_mask], data.y[data.test_mask])
    pred = output[data.test_mask].argmax(dim=1)
    acc = accuracy(pred, data.y[data.test_mask])
    
    
    return loss, acc

def run_exp(config=None):
     with wandb.init(config=config, project="IterativeMethods", job_type="exp_param"):
        config = wandb.config

        data, num_features, num_classes = make_data(config)
        iterative_model, normal_model = make_models(config, 
                                               input_dim=num_features, 
                                               output_dim=num_classes)
        iterative_model_param = count_parameters(iterative_model)
        normal_model_param = count_parameters(normal_model)
        wandb.log({
            'i_num_param': iterative_model_param,
            'n_num_param': normal_model_param
        })
        
        train(iterative_model, data, config, config.num_epochs, iter=True)
        loss_iter, acc_iter = test(iterative_model, data)
        wandb.log({
        'iter_test_loss': loss_iter,
        'iter_test_accuracy': acc_iter
        })
        train(normal_model, data, config, config.num_epochs, iter=False)
        loss_normal, acc_normal = test(normal_model, data)
        wandb.log({
        'normal_test_loss': loss_normal,
        'normal_test_accuracy': acc_normal
        })

config = {
    'num_epochs': 200,
    'dataset_name': "Cora",
    'noise_percent': 0.7,
    'hid_dim': 32,
    'num_iterations': 10,
    'num_layers': 10,
    'smooth_fac': 0.7,
    'dropout': 0.5,
    'learning_rate': 0.01,
    'weight_decay': 5e-4
} 

run_exp(config)