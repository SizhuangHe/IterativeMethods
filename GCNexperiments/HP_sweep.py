from __future__ import division
from __future__ import print_function
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from utils import accuracy
from models import iterativeGCN

import wandb
wandb.login()

def build_model(config, input_dim, output_dim):
    model = iterativeGCN(input_dim=input_dim,
                            output_dim=output_dim,
                            hidden_dim=config.hid_dim,
                            num_train_iter=config.num_iterations,
                            smooth_fac=config.smooth_fac,
                            dropout=config.dropout)
    return model

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
    
    return loss, acc

def run_exp(config=None):
    with wandb.init(config=config):
        config = wandb.config

        dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
        data = dataset[0]
        input_dim = dataset.num_features
        output_dim = dataset.num_classes
        model = build_model(config, input_dim, output_dim)
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        for epoch in range(config.num_epochs):
            loss_train, acc_train = train_epoch(model, data, optimizer)
            loss_val, acc_val = validate_epoch(model, data)
            wandb.log({
                'training_loss': loss_train,
                'training_accuracy': acc_train,
                'validation_loss': loss_val,
                'validation_accuracy': acc_val,
                "epoch": epoch+1
            })
        
        loss_test, acc_test = test(model, data)
        wandb.log({
                'test_loss': loss_test,
                'test_accuracy': acc_test
        })

sweep_config = {
    'method': 'random'
}

metric = {
    'name': 'accuracy',
    'goal': 'maximize'
}
sweep_config['metric'] = metric

parameters_dict = {
    'num_iterations': {
        'values':[2, 3]
    },
    'learning_rate': {
        'values': np.arange(0.0005, 0.02, 0.0005).tolist()
    },
    'smooth_fac': {
        'values': np.arange(0.1, 1, 0.05).tolist()
    },
    'hid_dim': {
        'values': [16, 32]
    },
    'weight_decay': {
        'values': [1e-4, 2e-4, 3e-4, 4e-4, 5e-4]
    },
    'num_runs': {
        'value': 5
    },
    'num_epochs': {
        'value': 200
    },
    'dropout': {
        'value': 0.5
    }
}
sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project="IterativeMethods")
wandb.agent(sweep_id, run_exp, count=50)
    
        