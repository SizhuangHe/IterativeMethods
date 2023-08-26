from __future__ import division
from __future__ import print_function
import numpy as np
import torch

import sys
from pathlib import Path
BASE_PATH = Path(__file__).parent.parent.parent.parent.absolute()
sys.path.insert(1, str(BASE_PATH))
from torch.optim import Adam
from torch.optim.lr_scheduler import ConstantLR
import torch.nn.functional as F

from src.utils.utils import build_iterativeGCN, make_Planetoid_data, exp_per_model, make_uniform_schedule
from src.models.iterativeModels import iterativeGCN_Planetoid
from src.utils.metrics import MAD
from argparse import ArgumentParser

import wandb
wandb.login()

parser = ArgumentParser()
parser.add_argument("--hid_dim", type=int, help="the hidden dimension of the model", default=32)
parser.add_argument("--num_epochs", type=int, help="number of training epochs", default=200)
parser.add_argument("--dataset", type=str, help="name of the Planetoid dataset, choose from Cora, CiteSeer and PubMed", 
                    choices=["Cora", "CiteSeer", "PubMed"], 
                    default="Cora")
parser.add_argument("--noise", type=float, help="the amount of noise to add to the dataset", choices=[0, 0.5, 0.7], default=0)
args = parser.parse_args()



def run_exp(config=None):
    wandb.init(job_type="Sweep", 
               project="IterativeMethods", 
               config=config, 
               notes="Fix noise, sweep for the best hyperparams",
               tags=["iterativeGCN"])
    config = wandb.config
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    wandb.log({
        "device": device_str
    })
    train_schedule = make_uniform_schedule(config.num_iter_layers, config.smooth_fac)
    wandb.log({
        "train_schedule": train_schedule
    })
    data, num_features, num_classes = make_Planetoid_data(config, seed=2147483647)
    
    
    model = build_iterativeGCN(config, num_features, num_classes, train_schedule)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = ConstantLR(optimizer, factor=1, total_iters=1)
    
    exp_per_model(model, data, optimizer, scheduler, config, device)
    
    data = data.to(device)
    out = model(data.x, data.edge_index).cpu()
    out = F.softmax(out)
    mad = MAD(out.detach())
    wandb.log({
        "MAD": mad
    })
    
    wandb.finish()
    
        
        
        

sweep_config = {
    'method': 'random'
}

metric = {
    'name': 'accuracy',
    'goal': 'maximize'
}
sweep_config['metric'] = metric

parameters_dict = {
    'num_iter_layers': {
        'values': [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    },
    'learning_rate': {
        'values': np.arange(0.001, 0.02, 0.001).tolist()
    },
    'smooth_fac': {
        'values': [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
    },
    'hid_dim': {
        'value': args.hid_dim
    },
    'weight_decay': {
        'values': [0, 0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005]
    },
    'num_epochs': {
        'value': args.num_epochs
    },
    'dropout': {
        'value': 0.5
    },
    'dataset_name': {
        'value': args.dataset
    },
    'noise_percent': {
        'value': args.noise
    }
}
sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project="IterativeMethods")
wandb.agent(sweep_id, run_exp, count=500)
    
        