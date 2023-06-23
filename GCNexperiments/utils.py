import numpy as np
import torch
import math
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from models import iterativeGCN

import wandb

'''
This file contains for utility functions for experiments.
'''

def accuracy(guess, truth):
    correct = guess == truth
    acc = correct.sum().item() / truth.size(dim=0)
    return acc

def make_uniform_schedule(length, smooth_fac):
    return np.full(length, smooth_fac)

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

def train(model, data, config):
    wandb.watch(model, log="all", log_freq=10)
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    for epoch in range(config.num_epochs):
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

def make_Planetoid_data(config):
    dataset = Planetoid(root='data/Planetoid', 
                        name=config['dataset_name'], 
                        transform=NormalizeFeatures())
    data = dataset[0]
    data = add_noise(data, percent=config['noise_percent'])
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    return data, num_features, num_classes

def exp_per_model(model, data, config):
    num_params = count_parameters(model)
    wandb.log({ 
            'num_param': num_params
    }) 
    train(model, data, config)
    loss_test, acc_test = test(model, data)
    wandb.log({
        'test_loss': loss_test,
        'test_accuracy': acc_test
    })

def build_iterativeGCN(config, input_dim, output_dim, train_schedule):
    model = iterativeGCN(input_dim=input_dim,
                            output_dim=output_dim,
                            hidden_dim=config.hid_dim,
                            dropout=config.dropout,
                            train_schedule=train_schedule,
                            xavier_init=True)
    return model