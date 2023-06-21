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


def add_noise(data, percent=0.7):
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

def make_data(dataset_dict):
    dataset = Planetoid(root='data/Planetoid', 
                        name=dataset_dict.dataset, 
                        transform=NormalizeFeatures())
    data = dataset[0]
    data = add_noise(data, percent=dataset_dict.noise_percent)
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    return data, num_features, num_classes

def make_iterative_model(model_dict, input_dim, output_dim):
    model = iterativeGCN(input_dim=input_dim,
                            output_dim=output_dim,
                            hidden_dim=model_dict.hid_dim,
                            num_train_iter=model_dict.num_iterations,
                            smooth_fac=model_dict.smooth_fac,
                            dropout=model_dict.dropout)
    return model

def make_normal_model(model_dict, input_dim, output_dim):
    model = GCN(input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=model_dict.hid_dim,
                num_layers=model_dict.num_layers,
                dropout=model_dict.dropout)
    return model

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

def test(model, data):
    model.eval()
    output = model(data.x, data.edge_index)
    loss = F.nll_loss(output[data.test_mask], data.y[data.test_mask])
    pred = output[data.test_mask].argmax(dim=1)
    acc = accuracy(pred, data.y[data.test_mask])
    wandb.log({
        'test_loss': loss,
        'test_accuracy': acc
    })
    
    return loss, acc

def train(model, data,optimizer_dict, num_epochs):
    optimizer = optim.Adam(model.parameters(), lr=optimizer_dict.learning_rate, weight_decay=optimizer_dict.weight_decay)
    for epoch in range(num_epochs):
        loss_train, acc_train = train_epoch(model, data, optimizer)
        loss_val, acc_val = validate_epoch(model, data)
        wandb.log({
            'training_loss': loss_train,
            'training_accuracy': acc_train,
            'validation_loss': loss_val,
            'validation_accuracy': acc_val,
            "epoch": epoch+1
        })


def run_exp(config=None):
     with wandb.init(config=config):
        config = wandb.config

        data, num_features, num_classes = make_data(config.dataset)
        iterative_model = make_iterative_model(config.iterative_model, 
                                               input_dim=num_features, 
                                               output_dim=num_classes)
        normal_model = make_normal_model(config.normal_model, 
                                         input_dim=num_features,
                                         output_dim=num_classes)
        iterative_model_param = count_parameters(iterative_model)
        normal_model_param = count_parameters(normal_model)
        wandb.log({
            'Number of parameters for the iterative model': iterative_model_param,
            'Number of parameters for the normal model': normal_model_param
        })
        
        train(iterative_model, data, config.optimizer, config.num_epochs)
        test(iterative_model, data)

config = {
    'num_epochs': 200
} 
dataset_dict = {
    'dataset': "Cora",
    'noise_percent': 0.7
}
config['dataset'] = dataset_dict
iterative_model_dict = {
    'hid_dim': 32,
    'num_train_iter': 2,
    'smooth_fac': 0.7,
    'dropout': 0.5
}
config['iterative_model'] = iterative_model_dict
normal_model_dict = {
    'hid_dim': 32,
    'num_layers': 2,
    'dropout': 0.5
}
config['normal_model'] = normal_model_dict
optimizer_dict = {
    'learning_rate': 0.01,
    'weight_decay': 5e-4
}
config['optimizer'] = optimizer_dict