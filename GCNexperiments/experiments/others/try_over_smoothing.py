import pandas as pd
import matplotlib.pyplot as plt
import wandb

import numpy as np
import torch.nn.functional as F

import sys
from pathlib import Path
BASE_PATH = Path(__file__).parent.parent.absolute()
sys.path.insert(1, str(BASE_PATH))

from src.utils.utils import make_Planetoid_data, exp_per_model, make_uniform_schedule
from src.utils.metrics import MAD
from src.models.iterativeModels import iterativeGCN
from src.models.GCN import GCN


def run_exp(hyper=None):
    data, num_features, num_classes = make_Planetoid_data(hyper)
    wandb.init(config=hyper, job_type="over_smoothing", project="IterativeMethods", tags=["iterativeGCN"])
    config = wandb.config
    train_schedule = make_uniform_schedule(config.num_iter_layers, config.smooth_fac)  
    wandb.log({
        'train_schedule': train_schedule
    })

    model = iterativeGCN(input_dim=num_features,
                                    output_dim=num_classes,
                                    hidden_dim=config.hid_dim,
                                    train_schedule=train_schedule,
                                    dropout=config.dropout)
    exp_per_model(model, data, config)

    out = model(data.x, data.edge_index)
    i_mad = MAD(F.softmax(out, dim=1))
    del model
    wandb.finish()

    wandb.init(job_type="over_smoothing", project="IterativeMethods", config=hyper, notes="usualGCN",tags=["usualGCN"])
    config = wandb.config
    model = GCN(input_dim=num_features,
                output_dim=num_classes,
                hidden_dim=config.hid_dim,
                num_layers=config.num_iter_layers,
                dropout=config.dropout,
                )
    exp_per_model(model, data, config)
    out2 = model(data.x, data.edge_index)
    u_mad = MAD(F.softmax(out2, dim=1))
    del model
    wandb.finish()
    print("MAD: iterative: {:.4f}, usual:{:.4f}".format(i_mad, u_mad))

hyper = {
    'num_epochs': 200,
    'dataset_name': "Cora",
    'noise_percent': 0,
    'hid_dim': 32,
    'num_iter_layers': 20,
    'smooth_fac': 0.8,
    'dropout': 0.5,
    'learning_rate': 0.01,
    'weight_decay': 4e-4
} 

run_exp(hyper=hyper)