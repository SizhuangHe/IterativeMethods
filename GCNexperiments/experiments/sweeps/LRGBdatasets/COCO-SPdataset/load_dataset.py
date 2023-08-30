from __future__ import division
from __future__ import print_function
import numpy as np
import torch

import sys
from pathlib import Path
BASE_PATH = Path(__file__).parent.parent.parent.parent.parent.absolute()
sys.path.insert(1, str(BASE_PATH))
import torch
from src.utils.utils import exp_vocsp, make_uniform_schedule
from src.models.iterativeModels import iterativeGCN_vocsp
from src.utils.metrics import MAD
from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR


import wandb
wandb.login()


train_dataset = LRGBDataset(root="/vast/palmer/scratch/dijk/sh2748/data/LRGB", name="COCO-SP", split="train")
val_dataset = LRGBDataset(root="/vast/palmer/scratch/dijk/sh2748/data/LRGB", name="COCO-SP", split="val")
test_dataset = LRGBDataset(root="/vast/palmer/scratch/dijk/sh2748/data/LRGB", name="COCO-SP", split="test")