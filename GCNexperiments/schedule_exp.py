from __future__ import division
from __future__ import print_function
import time
import numpy as np
import math
import torch
import logging
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from utils import run_experiment
from models import GCN, iterativeGCN


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s: %(message)s")
file_handler = logging.FileHandler("schedule_exp.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

schedule = F.sigmoid(torch.Tensor(np.arange(1, 10)))
lr = 0.01
weight_decay = 5e-4


dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
data = dataset[0]

model = iterativeGCN(input_dim=dataset.num_features,
                                        output_dim=dataset.num_classes,
                                        hidden_dim=16,
                                        num_train_iter=2,
                                        smooth_fac=0.7,
                                        schedule=schedule,
                                        dropout=0.5,
                                        xavier_init=True
                                        )

loss_test, acc_test, training_time = run_experiment(model=model,
                                                    data=data,
                                                    lr=lr,
                                                    weight_decay=weight_decay,
                                                    model_name="lala",
                                                    run=1,
                                                    num_epochs=200
                                                    )
print(acc_test)