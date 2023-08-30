import numpy as np
import torch
import math
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import add_remaining_self_loops
from src.models.iterativeModels import iterativeGCN_mol, iterativeGCN_Planetoid
from ogb.graphproppred import Evaluator
from tqdm import tqdm
from math import sqrt
from sklearn.metrics import f1_score, average_precision_score
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from src.utils.metrics import eval_ap

import wandb

'''
This file contains utilities for experiments.
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

def train_epoch(model, data, optimizer, scheduler, device):
    model.train()
    data = data.to(device)
    
    output = model(data.x, data.edge_index)
    loss = F.cross_entropy(output[data.train_mask], data.y[data.train_mask])
    pred = F.log_softmax(output[data.train_mask], dim=1).argmax(dim=1)
    acc = accuracy(pred, data.y[data.train_mask])
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    wandb.log({
            "lr_scheduler": scheduler.get_last_lr()[0]
        })
    return loss, acc

def validate_epoch(model, data, device):
    model.eval()
    data = data.to(device)

    output = model(data.x, data.edge_index)
    loss = F.cross_entropy(output[data.val_mask], data.y[data.val_mask])
    pred = F.log_softmax(output[data.val_mask], dim=1).argmax(dim=1)
    acc = accuracy(pred, data.y[data.val_mask])
    return loss, acc

def train(model, data, optimizer, scheduler, config, device):
    wandb.watch(model, log="all", log_freq=10)
    
    for epoch in range(config.num_epochs):
        loss_train, acc_train = train_epoch(model, data, optimizer, scheduler, device)
        loss_val, acc_val = validate_epoch(model, data, device)

        wandb.log({
            'training_loss': loss_train,
            'training_accuracy': acc_train,
            'validation_loss': loss_val,
            'validation_accuracy': acc_val,
            "epoch": epoch+1,
        })
  
def test(model, data, device):
    model.eval()
    data = data.to(device)

    output = model(data.x, data.edge_index)
    loss = F.cross_entropy(output[data.test_mask], data.y[data.test_mask])
    pred = F.log_softmax(output[data.test_mask], dim=1).argmax(dim=1)
    acc = accuracy(pred, data.y[data.test_mask])
    
    return loss, acc

def exp_per_model(model, data, optimizer, scheduler, config, device):
    num_params = count_parameters(model)
    wandb.log({ 
            'num_param': num_params
    }) 
    train(model, data, optimizer, scheduler, config, device)
    loss_test, acc_test = test(model, data, device)
    wandb.log({
        'test_loss': loss_test,
        'test_accuracy': acc_test
    })

def train_mol_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()
    epoch_loss = 0
    for step, batched_data in enumerate(loader):  # Iterate in batches over the training dataset.
        batched_data = batched_data.to(device)
        pred = model(batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch)
        ## ignore nan targets (unlabeled) when computing training loss.
        is_labeled = batched_data.y == batched_data.y
        loss = criterion(pred.to(torch.float32)[is_labeled], batched_data.y.to(torch.float32)[is_labeled])
        epoch_loss += loss.item()
        optimizer.zero_grad()  
        loss.backward() 
        optimizer.step()
        scheduler.step()
        wandb.log({
            "lr_scheduler": scheduler.get_last_lr()[0]
        })
    wandb.log({
        "training_loss": epoch_loss
    })
          
def eval_mol(model, loader, evaluator, device):
    model.eval()
    y_true = []
    y_pred = []
    for step, batched_data in enumerate(loader):
        batched_data = batched_data.to(device)
        with torch.no_grad():
            pred = model(batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch)
            y_true.append(batched_data.y.view(pred.shape).detach())
            y_pred.append(pred.detach())
    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_pred = torch.cat(y_pred, dim = 0).cpu().numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    return evaluator.eval(input_dict)

def train_mol(model, optimizer, scheduler, train_loader, valid_loader, evaluator,config, device):
    wandb.watch(model, log="all", log_freq=1000)
    
    for epoch in range(config.num_epochs):
        train_mol_epoch(model, train_loader, optimizer, scheduler, device)
        val_ap = eval_mol(model, valid_loader, evaluator, device)
        train_ap = eval_mol(model, train_loader, evaluator, device)
        wandb.log({
            "Validate ap": val_ap,
            "Train ap": train_ap,
            "epoch": epoch+1
        })

def exp_mol(model, optimizer, scheduler,train_loader, valid_loader, test_loader, evaluator,config, device):
    num_params = count_parameters(model)
    wandb.log({ 
            'num_param': num_params
    }) 
    train_mol(model, optimizer, scheduler,train_loader, valid_loader, evaluator, config, device)
    test_ap=eval_mol(model, test_loader, evaluator, device)
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

def build_iterativeGCN(config, input_dim, output_dim, train_schedule):
    model = iterativeGCN_Planetoid(input_dim=input_dim,
                            output_dim=output_dim,
                            hidden_dim=config.hid_dim,
                            dropout=config.dropout,
                            train_schedule=train_schedule,
                            xavier_init=True)
    return model

def train_arxiv_epoch(model, data, train_idx, optimizer, scheduler, device):
    model.train()
    data = data.to(device)

    out = model(data.x, data.adj_t)[train_idx]
    loss = F.cross_entropy(out, data.y.squeeze(1)[train_idx])
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    wandb.log({
            "lr_scheduler": scheduler.get_last_lr()[0]
    })
    
    epoch_loss = loss.item()
    return epoch_loss

def eval_arxiv(model, data, idx, evaluator, device):
    model.eval()
    data = data.to(device)

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True).cpu()
    y_true = data.y.cpu()
    acc = evaluator.eval({
        'y_true': y_true[idx],
        'y_pred': y_pred[idx],
    })['acc']
    return acc

def train_arxiv(model, data, optimizer, train_idx, valid_idx, evaluator, scheduler, num_epochs, device):
    wandb.watch(model, log="all", log_freq=10)
    
    for epoch in range(num_epochs):
        train_loss = train_arxiv_epoch(model, data, train_idx, optimizer,scheduler, device)
        train_acc = eval_arxiv(model, data, valid_idx, evaluator, device)
        valid_acc = eval_arxiv(model, data, train_idx, evaluator, device)
        wandb.log({
            'training_loss': train_loss,
            'training_accuracy': train_acc,
            'validation_accuracy': valid_acc,
            "epoch": epoch+1
        })

def exp_arxiv(model, data, optimizer, scheduler, train_idx, valid_idx, test_idx, evaluator, num_epochs, device):
    num_params = count_parameters(model)
    wandb.log({ 
            'num_param': num_params
    })
    train_arxiv(model, data, optimizer, train_idx, valid_idx, evaluator, scheduler, num_epochs, device)
    test_acc = eval_arxiv(model, data, test_idx, evaluator, device)
    wandb.log({
        'test_accuracy': test_acc
    })

def weighted_cross_entropy(pred, true):
    """Weighted cross-entropy for unbalanced classes.
    """
    
    # calculating label weights for weighted loss computation
    V = true.size(0)
    
    n_classes = pred.shape[1] if pred.ndim > 1 else 2
    label_count = torch.bincount(true)
    label_count = label_count[label_count.nonzero(as_tuple=True)].squeeze()
    cluster_sizes = torch.zeros(n_classes, device=pred.device).long()
    cluster_sizes[torch.unique(true)] = label_count
    weight = (V - cluster_sizes).float() / V
    weight *= (cluster_sizes > 0).float()

    # multiclass
    if pred.ndim > 1:
        pred = F.log_softmax(pred, dim=-1)
        loss = F.nll_loss(pred, true, weight=weight)
        
        return loss
    # binary
    else:
        loss = F.binary_cross_entropy_with_logits(pred, true.float(),
                                                    weight=weight[true])
        return loss

def train_vocsp_epoch(model, loader, optimizer, scheduler, device='cpu'):
    model.train()
    criterion = weighted_cross_entropy
    epoch_loss = 0
    for batched_data in loader:  # Iterate in batches over the training dataset.
        batched_data = batched_data.to(device)
        pred = model(batched_data.x, batched_data.edge_index, batched_data.edge_attr,batched_data.batch) # size of pred is [number of nodes, number of features]
        true = batched_data.y
        loss = criterion(pred, true)
        epoch_loss += loss.item()
        optimizer.zero_grad()  
        loss.backward() 
        optimizer.step()
        if isinstance(scheduler, OneCycleLR):
            scheduler.step()
        
    return epoch_loss

def eval_vocsp(model, loader, device='cpu'):
    model.eval()
    y_true = []
    y_pred = [] 
    
    criterion = weighted_cross_entropy
    val_loss = 0
    for batched_data in loader:  # Iterate in batches over the training dataset.
        batched_data = batched_data.to(device)
        pred = model(batched_data.x, batched_data.edge_index, batched_data.edge_attr,batched_data.batch) # size of pred is [number of nodes, number of features]
        true = batched_data.y
        loss = criterion(pred, true)
        val_loss += loss.item()

        pred_val = pred.max(dim=1)[1] # pred_val contains actually class predictions
        y_pred.append(pred_val.detach())
        y_true.append(true.detach())

    
    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_pred = torch.cat(y_pred, dim = 0).cpu().numpy()
    val_f1 = f1_score(y_true, y_pred, average="macro")
        
    return val_loss, val_f1

def train_vocsp(model, optimizer, scheduler, train_loader, valid_loader, num_epochs, device='cpu'):
    wandb.watch(model, log="all", log_freq=10)
    for epoch in range(num_epochs):
        train_loss = train_vocsp_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_f1= eval_vocsp(model, valid_loader, device)
        
        wandb.log({
            "Train loss": train_loss,
            "Validate f1": val_f1,
            "Validate loss": val_loss,
            "epoch": epoch+1,
            "lr": optimizer.param_groups[0]['lr']
        })
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        

def exp_vocsp(model, optimizer, scheduler,train_loader, valid_loader, test_loader, num_epochs, device):
    num_params = count_parameters(model)
    wandb.log({ 
            'num_param': num_params
    }) 
    train_vocsp(model, optimizer, scheduler, train_loader, valid_loader, num_epochs, device)
    test_loss, test_f1=eval_vocsp(model, test_loader, device)
    wandb.log({
        "Test loss": test_loss,
        "Test f1": test_f1
    })
    
def train_pepfunc_epoch(model, loader, optimizer, scheduler, device='cpu'):
    model.train()
    criterion = F.cross_entropy
    epoch_loss = 0
    for batched_data in loader:  # Iterate in batches over the training dataset.
        batched_data = batched_data.to(device)
        pred = model(batched_data.x, batched_data.edge_index, batched_data.edge_attr,batched_data.batch) # size of pred is [number of nodes, number of features]
        true = batched_data.y
        loss = criterion(pred, true)
        epoch_loss += loss.item()
        optimizer.zero_grad()  
        loss.backward() 
        optimizer.step()
        if isinstance(scheduler, OneCycleLR):
            scheduler.step()
        
    return epoch_loss

def eval_pepfunc(model, loader, device='cpu'):
    model.eval()
    y_true = []
    y_pred = [] 
    
    criterion = weighted_cross_entropy
    val_loss = 0
    for step, batched_data in enumerate(loader):  # Iterate in batches over the training dataset.
        batched_data = batched_data.to(device)
        pred = model(batched_data.x, batched_data.edge_index, batched_data.edge_attr,batched_data.batch) 
        true = batched_data.y
        loss = criterion(pred, true)
        val_loss += loss.item()

    
        y_pred.append(pred.detach())
        y_true.append(true.detach())

    
    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_pred = torch.cat(y_pred, dim = 0).cpu().numpy()
    val_ap = eval_ap(y_true, y_pred)
      
    return val_loss, val_ap

def train_pepfunc(model, optimizer, scheduler, train_loader, valid_loader, num_epochs,device='cpu'):
    wandb.watch(model, log="all", log_freq=10)
    for epoch in range(num_epochs):
        train_loss = train_pepfunc_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_ap = eval_pepfunc(model, valid_loader, device)
        
        wandb.log({
            "Train loss": train_loss,
            "Validate loss": val_loss,
            "Validate AP": val_ap,
            "epoch": epoch+1,
            "lr": optimizer.param_groups[0]['lr']
        })
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        

def exp_pepfunc(model, optimizer, scheduler,train_loader, valid_loader, test_loader, num_epochs, device):
    num_params = count_parameters(model)
    wandb.log({ 
            'num_param': num_params
    }) 
    train_pepfunc(model, optimizer, scheduler, train_loader, valid_loader, num_epochs, device)
    test_loss, test_ap=eval_pepfunc(model, test_loader, device)
    wandb.log({
        "Test loss": test_loss,
        "Test AP": test_ap
    })
    
       