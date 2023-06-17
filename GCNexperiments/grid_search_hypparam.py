from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from  utils import run_experiment
from models import iterativeGCN

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s: %(message)s")
file_handler = logging.FileHandler("grid_search.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


DS_NAME = ['Cora'] #only Cora for now
LR = np.arange(0.0115, 0.02, 0.0005)
SMOOTH_FAC = np.arange(0.5, 1 , 0.1) 
HID_DIM = [8, 16, 32] # 3
WD = [1e-4, 3e-4, 5e-4]
num_runs = 3

# total_experiments =  len(DS_NAME) * LR.size * SMOOTH_FAC.size * len(HID_DIM) * len(WD) *  num_runs
total_experiments =  len(DS_NAME) * len(LR) * len(SMOOTH_FAC) * len(HID_DIM) * len(WD) *  num_runs
print("{} sets of parameters to search from!".format(total_experiments))

for ds_name in DS_NAME:
    logger.info("Experiment begins, dataset: {}".format(ds_name))
    
    dataset = Planetoid(root='data/Planetoid', name=ds_name, transform=NormalizeFeatures())
    data = dataset[0]
    
    running_max_acc = 0
    max_lr = -1
    max_sm_fac = -1
    max_weight_decay = -1
    max_hid_dim = -1
    curr_exp = 1
    num_param_set = 1
   
    for lr in LR:
        for smooth_fac in SMOOTH_FAC:
            for hid_dim in HID_DIM:
                for weight_decay in WD:
                    run_acc = 0
                    for run in range(num_runs):
                        model = iterativeGCN(input_dim=dataset.num_features,
                                                output_dim=dataset.num_classes,
                                                hidden_dim=16,
                                                num_train_iter=2,
                                                num_eval_iter=2,
                                                smooth_fac=smooth_fac,
                                                dropout=0.5)
                        loss_test, acc_test, training_time = run_experiment(model=model, 
                                                                            data=data, 
                                                                            lr=lr, 
                                                                            weight_decay=2e-4,
                                                                            model_name=str(lr) + "_" + str(smooth_fac),
                                                                            run=run,
                                                                            num_epochs=200,
                                                                            )
                        run_acc += acc_test
                        print("Experiment {:04d}/{:04d} run {}/{} finished, time elapsed: {:.4}".format(curr_exp, total_experiments, run+1, num_runs, training_time))
                        curr_exp += 1 
                        del model
                
                    mean_acc = run_acc / num_runs
                    
                    logger.info("Parameter set {:04d}: hidden_dim = {:02d}, lr={:.4f}, smooth_fac={:.4f}, weight_decay = {:.4f}, mean test accuracy: {:.4}".format(num_param_set, hid_dim, lr, smooth_fac, weight_decay, mean_acc))
                    num_param_set += 1
            
                    if mean_acc > running_max_acc:
                        running_max_acc = mean_acc
                        max_lr = lr
                        max_sm_fac = smooth_fac
                        max_hid_dim = hid_dim
                        max_weight_decay = weight_decay
            
    logger.info("--> Experiment finished. Best accuracy: {:.4f}, parameters: lr={:.4f}, smooth_fac={:.4f}, hid_dim={}, weight decay={:.2e}".format(running_max_acc, max_lr, max_sm_fac, max_hid_dim, max_weight_decay))
    print("--> Experiment finished. Best accuracy: {:.4f}, parameters: lr={:.4f}, smooth_fac={:.4f}, hid_dim={}, weight decay={:.2e}".format(running_max_acc, max_lr, max_sm_fac, max_hid_dim, max_weight_decay))