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
LR = np.arange(0.001, 0.01, 0.0005)
SMOOTH_FAC = np.arange(0, 1 , 0.1)
WD = np.arange(1e-4, 1e-3, 1e-4)
HID_DIM = [8, 16, 32, 64]
num_runs = 3


for ds_name in DS_NAME:
    logger.info("Experiment begins, dataset: {}".format(ds_name))
    
    dataset = Planetoid(root='data/Planetoid', name=ds_name, transform=NormalizeFeatures())
    data = dataset[0]
    
    running_max_acc = 0
    max_lr = -1
    max_sm_fac = -1
    curr_exp = 1
    
    for lr in LR:
        for smooth_fac in SMOOTH_FAC:
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
                print("Experiment ", curr_exp,
                      "run ", run + 1,
                      ", total time: {:.4}".format(training_time),
                        "learning rate: {:.4},".format(lr), 
                        "smoothing factor: {:.4},".format(smooth_fac), 
                        "test accuracy: {:.4}".format(acc_test))
                del model
            
            mean_acc = run_acc / num_runs
            print("Experiment ", curr_exp, " summary: mean test accuracy:{:.4}".format(mean_acc))
            print()
            logger.info("Parameter set 1: hidden_dim = {:.4}, lr={:.4f}, smooth_fac={:.4f}, weight_decay = {:.4f}")
            curr_exp += 1    
            
            if mean_acc > running_max_acc:
                running_max_acc = mean_acc
                max_lr = lr
                max_sm_fac = smooth_fac
            
            

    print("Summary:")
    print("Best test accuracy: {:.4} ".format(running_max_acc),
           ". Hyperparameters: learning rate: {:.4} ".format(max_lr),
             ", smoothing factor: {:.4}".format(max_sm_fac))
    print()