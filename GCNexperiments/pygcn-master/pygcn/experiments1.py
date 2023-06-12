from __future__ import division
from __future__ import print_function

import time
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, run_experiment, print_stats
from models import GCN_2, GCN_3, ite_GCN

import os

# In this file, train/validation/test data partition is exactly the same as in the GCN paper.

adj, features, labels = load_data(path="../data/cora/", dataset="cora")
idx_train = range(140)
idx_val = range(200, 500)
idx_test = range(500, 1500)
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

hidden = 16
dropout = 0.5
lr = 0.01
weight_decay = 5e-4
num_epochs = 200
num_runs = 5 #TODO: change this

exp_directory = "../experiment_results"

GCN_2_loss = []
GCN_2_acc = []
GCN_2_time = []
for i in range(num_runs):
    # run 2L, 3L ten times each
    model = GCN_2(nfeat=features.shape[1],
            nhid=hidden,
            nclass=labels.max().item() + 1,
            dropout=dropout)
    loss, acc, train_time = run_experiment(num_epochs=num_epochs, model=model, lr=0.01, 
                   weight_decay=weight_decay, features=features, adj=adj, idx_train=idx_train, 
                   idx_val=idx_val, idx_test=idx_test, labels=labels, model_name = "2L", run=i)
    GCN_2_loss.append(loss)
    GCN_2_acc.append(acc)
    GCN_2_time.append(train_time)
    del model

GCN_3_loss = []
GCN_3_acc = []
GCN_3_time = []
for i in range(num_runs):
    model = GCN_3(nfeat=features.shape[1],
            nhid=hidden,
            nclass=labels.max().item() + 1,
            dropout=dropout)
    loss, acc, train_time = run_experiment(num_epochs=num_epochs, model=model, lr=0.01, 
                   weight_decay=weight_decay, features=features, adj=adj, idx_train=idx_train, 
                   idx_val=idx_val, idx_test=idx_test, labels=labels, model_name="3L", run=i)
    GCN_3_loss.append(loss)
    GCN_3_acc.append(acc)
    GCN_3_time.append(train_time)
    del model
    
    # TODO: process results, store in a array or something, write to file, 
    #       suppress verbose, keep all losses/accs as arrays


lr = 0.002
weight_decay = 1e-4
smooth_fac = 0.8

ite_2T_loss = []
ite_2T_acc = []
ite_2T_time = []
for i in range(num_runs):
    model = ite_GCN(nfeat=features.shape[1],
            nclass=labels.max().item() + 1,
            dropout=dropout,
            train_nite= 2,
            eval_nite= 0,
            allow_grad=True,
            smooth_fac=smooth_fac)
    loss, acc, train_time = run_experiment(num_epochs=num_epochs, model=model, lr=lr, 
               weight_decay=weight_decay, features=features, adj=adj, idx_train=idx_train, 
               idx_val=idx_val, idx_test=idx_test, labels=labels, model_name="2iT", run=i)
    ite_2T_loss.append(loss)
    ite_2T_acc.append(acc)
    ite_2T_time.append(train_time)
    del model


ite_3T_loss = []
ite_3T_acc = []
ite_3T_time = []
for i in range(num_runs):
    model = ite_GCN(nfeat=features.shape[1],
            nclass=labels.max().item() + 1,
            dropout=dropout,
            train_nite= 3,
            eval_nite= 0,
            allow_grad=True,
            smooth_fac=smooth_fac)
    loss, acc, train_time = run_experiment(num_epochs=num_epochs, model=model, lr=lr, 
                   weight_decay=weight_decay, features=features, adj=adj, idx_train=idx_train, 
                   idx_val=idx_val, idx_test=idx_test, labels=labels, model_name="3iT", run=i)
    ite_3T_loss.append(loss)
    ite_3T_acc.append(acc)
    ite_3T_time.append(train_time)
    del model

ite_2F_loss = []
ite_2F_acc = []
ite_2F_time = []
for i in range(num_runs):
    model = ite_GCN(nfeat=features.shape[1],
            nclass=labels.max().item() + 1,
            dropout=dropout,
            train_nite= 2,
            eval_nite= 0,
            allow_grad=False,
            smooth_fac=smooth_fac)
    loss, acc, train_time = run_experiment(num_epochs=num_epochs, model=model, lr=lr, 
                   weight_decay=weight_decay, features=features, adj=adj, idx_train=idx_train, 
                   idx_val=idx_val, idx_test=idx_test, labels=labels, model_name="2iF", run=i)
    ite_2F_loss.append(loss)
    ite_2F_acc.append(acc)
    ite_2F_time.append(train_time)
    del model

ite_3F_loss = []
ite_3F_acc = []
ite_3F_time = []
for i in range(num_runs):
    model = ite_GCN(nfeat=features.shape[1],
            nclass=labels.max().item() + 1,
            dropout=dropout,
            train_nite= 3,
            eval_nite= 0,
            allow_grad=False,
            smooth_fac=smooth_fac)
    loss, acc, train_time = run_experiment(num_epochs=num_epochs, model=model, lr=lr, 
                   weight_decay=weight_decay, features=features, adj=adj, idx_train=idx_train, 
                   idx_val=idx_val, idx_test=idx_test, labels=labels, model_name="3iF", run=i)
    ite_3F_loss.append(loss)
    ite_3F_acc.append(acc)
    ite_3F_time.append(train_time)
    del model

print_stats(model_name="Non-iterative 2 layer", acc_test=GCN_2_acc, training_time=GCN_2_time)
print_stats(model_name="Non-iterative 3 layer", acc_test=GCN_3_acc, training_time=GCN_3_time)
print_stats(model_name="Iterative 2 iteration, gradient: T", acc_test=ite_2T_acc, training_time=ite_2T_time)
print_stats(model_name="Iterative 3 iteration, gradient: T", acc_test=ite_3T_acc, training_time=ite_3T_time)
print_stats(model_name="Iterative 2 iteration, gradient: F", acc_test=ite_2F_acc, training_time=ite_2F_time)
print_stats(model_name="Iterative 3 iteration, gradient: F", acc_test=ite_3F_acc, training_time=ite_3F_time)
