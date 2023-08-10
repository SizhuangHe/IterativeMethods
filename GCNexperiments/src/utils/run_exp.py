import torch
import torch.nn.functional as F
from src.utils.metrics import MAD
from src.models.models import GCN_arxiv, GCN
from src.models.iterativeModels import iterativeGCN_arxiv, iterativeGAT
from src.models.variantModels import iterativeGCNv_arxiv
from src.utils.utils import exp_arxiv, make_uniform_schedule, make_Planetoid_data, exp_per_model
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import Adam
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import torch_geometric.transforms as T
import wandb
wandb.login()

def run_arxiv(config=None):
    wandb.init(job_type="obgn-arxiv",
               project="IterativeMethods", 
               config=config, 
               notes="uGCN",
               tags=["uGCN"])
    config = wandb.config
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', transform=T.ToSparseTensor())
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    data = dataset[0] # pyg graph object

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    wandb.log({
        "device": device_str
    })

    model = GCN_arxiv(input_dim=data.num_features,
                      hid_dim=config.hid_dim,
                      output_dim=dataset.num_classes,
                      num_layers=config.num_iter_layers,
                      dropout=config.dropout).to(device)
    evaluator = Evaluator(name="ogbn-arxiv")
    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = OneCycleLR(optimizer, max_lr=config.learning_rate, steps_per_epoch=1, epochs=config.num_epochs, pct_start=config.warmup_pct)
    exp_arxiv(model, data, optimizer, scheduler, train_idx, valid_idx, test_idx, evaluator, config.num_epochs, device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    out = model(data.x, data.adj_t).cpu()
    out = F.softmax(out, dim=1) # used raw out previously
    mad = MAD(out)
    wandb.log({
        "MAD": mad
    })

    wandb.finish()

def run_arxiv_iGCN(config=None):
    wandb.init(job_type="obgn-arxiv",
               project="IterativeMethods", 
               config=config, 
               notes="iGCN",
               tags=["iGCN"])
    config = wandb.config
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', transform=T.ToSparseTensor())
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    data = dataset[0] # pyg graph object

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    wandb.log({
        "device": device_str
    })
    train_schedule = make_uniform_schedule(config.num_iter_layers, config.smooth_fac)
    model = iterativeGCN_arxiv(input_dim=data.num_features,
                         output_dim=dataset.num_classes,
                         hidden_dim=config.hid_dim,
                         dropout=config.dropout,
                         train_schedule=train_schedule).to(device)
    evaluator = Evaluator(name="ogbn-arxiv")
    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = OneCycleLR(optimizer, max_lr=config.learning_rate, steps_per_epoch=1, epochs=config.num_epochs, pct_start=config.warmup_pct)
    exp_arxiv(model, data, optimizer, scheduler, train_idx, valid_idx, test_idx, evaluator, config.num_epochs, device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    out = model(data.x, data.adj_t).cpu()
    out = F.softmax(out, dim=1) # used raw out previously
    mad = MAD(out)
    wandb.log({
        "MAD": mad
    })

    wandb.finish() 

def run_arxiv_iGCNv(config=None):
    wandb.init(job_type="obgn-arxiv",
               project="IterativeMethods", 
               config=config, 
               notes="vGCN",
               tags=["vGCN"])
    config = wandb.config
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', transform=T.ToSparseTensor())
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    data = dataset[0] # pyg graph object

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    wandb.log({
        "device": device_str
    })
    train_schedule = make_uniform_schedule(config.num_iter_layers, config.smooth_fac)
    model = iterativeGCNv_arxiv(input_dim=data.num_features,
                         output_dim=dataset.num_classes,
                         hidden_dim=config.hid_dim,
                         dropout=config.dropout,
                         train_schedule=train_schedule).to(device)
    evaluator = Evaluator(name="ogbn-arxiv")
    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = OneCycleLR(optimizer, max_lr=config.learning_rate, steps_per_epoch=1, epochs=config.num_epochs, pct_start=config.warmup_pct)
    exp_arxiv(model, data, optimizer, scheduler, train_idx, valid_idx, test_idx, evaluator, config.num_epochs, device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    out = model(data.x, data.adj_t).cpu()
    out = F.softmax(out, dim=1) # used raw out previously
    mad = MAD(out)
    wandb.log({
        "MAD": mad
    })

    wandb.finish() 

def run_PM_iGAT(config=None):
    wandb.init(job_type="Sweep", 
               project="IterativeMethods", 
               config=config, 
               notes="Sweep for the iGAT",
               tags=["iterativeGAT"])
    
    config = wandb.config
    data, num_features, num_classes = make_Planetoid_data(config, seed=2147483647)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    wandb.log({
        "device": device_str
    })
    train_schedule = make_uniform_schedule(config.num_iter_layers, config.smooth_fac)
   
    model =iterativeGAT(input_dim=num_features,
                output_dim=num_classes,
                hidden_dim=config.hid_dim,
                train_schedule=train_schedule,
                heads=8,
                dropout=config.dropout,
                attn_dropout_rate=config.dropout,
                xavier_init=True
    ).to(device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = OneCycleLR(optimizer, max_lr=config.learning_rate, steps_per_epoch=1, epochs=config.num_epochs, pct_start=config.warmup_pct)
    exp_per_model(model=model, data=data, optimizer=optimizer, scheduler=scheduler, config=config, device=device)

    data = data.to(device)
    out = model(data.x, data.edge_index).cpu()
    out = F.softmax(out)
    mad = MAD(out.detach())
    wandb.log({
        "MAD": mad
    })

    wandb.finish()  