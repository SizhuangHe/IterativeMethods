import ogb
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.loader import DataLoader
import torch
import wandb
wandb.login()

import sys
from pathlib import Path
BASE_PATH = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(1, str(BASE_PATH))
from src.models.models import GCN_inductive
from src.utils.utils import exp_mol

dataset = PygGraphPropPredDataset(name="ogbg-molpcba") 
split_idx = dataset.get_idx_split() 
train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True)
valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False)
test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False)
evaluator = Evaluator(name="ogbg-molpcba")

def run_exp(config=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    wandb.init(job_type="molpcba",
               project="IterativeMethods", 
               config=config, 
               notes="usualGCN",
               tags=["usualGCN"])
    config = wandb.config
    model = GCN_inductive(
            num_tasks=dataset.num_tasks,
            hidden_dim=32,
            num_layers=2,
            dropout=0.5)
    model = model.to(device)
    exp_mol(model, train_loader, valid_loader, test_loader, evaluator, config, device)
    
    


config = {
    'num_epochs': 2,
    'dataset_name': "Cora",
    'noise_percent': 0,
    'hid_dim': 32,
    'num_iter_layers': 5,
    'smooth_fac': 0.6,
    'dropout': 0.5,
    'learning_rate': 0.005,
    'weight_decay': 4e-4
} 

run_exp(config)