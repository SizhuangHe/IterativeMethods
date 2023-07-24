import numpy as np
import torch
import math
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import add_remaining_self_loops
from src.models.iterativeModels import iterativeGCN
from ogb.graphproppred import Evaluator
from tqdm import tqdm

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

def add_noise(data, percent=0, seed=None):
    #add random 1's to data
    if percent > 0 and percent <= 1:
        len = np.prod(list(data.x.shape))
        ones = math.floor(len * percent)
        zeros = len - ones
        noise = torch.cat((torch.zeros(zeros), torch.ones(ones)))
        if seed is not None:
            rng = torch.Generator()
            rng.manual_seed(seed)
            noise = noise[torch.randperm(noise.size(0), generator=rng)]
        else:
            noise = noise[torch.randperm(noise.size(0))]
        noise = torch.reshape(noise, data.x.shape)
        data.x += noise
    return data

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_epoch(model, data, optimizer):
    model.train()
    output = model(data.x, data.edge_index)
    loss = F.cross_entropy(output[data.train_mask], data.y[data.train_mask])
    pred = F.log_softmax(output[data.train_mask], dim=1).argmax(dim=1)
    acc = accuracy(pred, data.y[data.train_mask])
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss, acc

def validate_epoch(model, data):
    model.eval()
    output = model(data.x, data.edge_index)
    loss = F.cross_entropy(output[data.val_mask], data.y[data.val_mask])
    pred = F.log_softmax(output[data.val_mask], dim=1).argmax(dim=1)
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
    loss = F.cross_entropy(output[data.test_mask], data.y[data.test_mask])
    pred = F.log_softmax(output[data.test_mask], dim=1).argmax(dim=1)
    acc = accuracy(pred, data.y[data.test_mask])
    
    return loss, acc

def train_mol_epoch(model, loader, optimizer):
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()
    for step, batched_data in enumerate(tqdm(loader, desc="Iteration")):  # Iterate in batches over the training dataset.
        
        pred = model(batched_data.x, batched_data.edge_index, batched_data.batch)
        ## ignore nan targets (unlabeled) when computing training loss.
        is_labeled = batched_data.y == batched_data.y
        loss = criterion(pred.to(torch.float32)[is_labeled], batched_data.y.to(torch.float32)[is_labeled])
        optimizer.zero_grad()  
        loss.backward() 
        optimizer.step()
        
def eval_mol(model, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []
    for step, batched_data in enumerate(tqdm(loader, desc="Iteration")):
        with torch.no_grad():
            pred = model(batched_data.x, batched_data.edge_index, batched_data.batch)
            y_true.append(batched_data.y.view(pred.shape).detach())
            y_pred.append(pred.detach())
    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    return evaluator.eval(input_dict)

def train_mol(model, train_loader, valid_loader, evaluator,config):
    wandb.watch(model, log="all", log_freq=10)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    for epoch in range(config.num_epochs):
        train_mol_epoch(model, train_loader, optimizer)
        ap = eval_mol(model, valid_loader, evaluator)
        wandb.log({
            "Validate ap": ap
        })

def test_mol(model, loader, evaluator):
    model.eval()
    ap = eval_mol(model, loader, evaluator)
    return ap

def exp_mol(model, train_loader, valid_loader, test_loader, evaluator,config):
    num_params = count_parameters(model)
    wandb.log({ 
            'num_param': num_params
    }) 
    train_mol(model, train_loader, valid_loader, evaluator)
    test_ap=test_mol(model, test_loader, evaluator)
    wandb.log({
        "Test ap": test_ap
    })

def make_Planetoid_data(config, seed=None):
    dataset = Planetoid(root='data/Planetoid', 
                        name=config['dataset_name'], 
                        transform=NormalizeFeatures(),
                        )
    data = dataset[0]
    data = add_noise(data, percent=config['noise_percent'], seed=seed)
    data.edge_index = add_remaining_self_loops(data.edge_index)[0]
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    return data, num_features, num_classes

def make_Amazon_data(config, seed=None):
    dataset = Amazon(root='data/Amazon',
                     name=config['dataset_name'],
                     transform=NormalizeFeatures())
    data = dataset[0]
    # data = add_noise(data, percent=config['noise_percent'], seed=seed)
    data.edge_index = add_remaining_self_loops(data.edge_index)[0]
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