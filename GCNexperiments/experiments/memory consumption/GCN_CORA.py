import torch
from torch.profiler import profile, record_function, ProfilerActivity
import sys
from pathlib import Path
BASE_PATH = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(1, str(BASE_PATH))
from src.utils.utils import make_Planetoid_data, exp_per_model
from src.models.models import GCN
from src.utils.metrics import MAD
import wandb
wandb.login()

config = {
    'num_epochs': 200,
    'dataset_name': "Cora",
    'noise_percent': 0,
    'hid_dim': 32,
    'num_iter_layers': 5,
    'smooth_fac': 0.7,
    'dropout': 0.5,
    'learning_rate': 0.01,
    'weight_decay': 4e-4
} 

def run_exp(config=None):
    wandb.init(job_type="run_GCN", 
               project="IterativeMethods", 
               config=config, 
               notes="usualGCN",
               tags=["usualGCN"])
    config = wandb.config
    data, num_features, num_classes = make_Planetoid_data(config)

    model = GCN(input_dim=num_features,
                                    output_dim=num_classes,
                                    hidden_dim=config.hid_dim,
                                    num_layers=config.num_iter_layers,
                                    dropout=config.dropout,
                                    )
    with profile(activities=[ProfilerActivity.CPU],
        profile_memory=True, record_shapes=True) as prof:
        model(data.x, data.edge_index)
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
    
run_exp(config)