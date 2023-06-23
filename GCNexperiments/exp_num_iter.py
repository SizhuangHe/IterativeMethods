from __future__ import division
from __future__ import print_function
import numpy as np
import torch
import torch.nn.functional as F
from utils import build_iterativeGCN, make_Planetoid_data, train, test
from models import iterativeGCN
import wandb
wandb.login()

'''
In this script, we train one iterativeGCN with a fixed schedule and do inference with another fixed schedule.
We take the output at each step of the iteration and pass it to the decoder to see how the performance change
over iterations.
'''

def run_exp(config=None, full_schedule=None, schedule_name=None):
    wandb.init(config=config, 
               project="IterativeMethods", 
               job_type="num_params", 
               tags=["iterativeGCN"], 
               notes="Train 1 model and look at it's performance at each iteration during inference")
    
    config = wandb.config
    data, num_features, num_classes = make_Planetoid_data(config)
    model = build_iterativeGCN(config, input_dim=num_features, output_dim=num_classes)
    train(model, data, config)
    loss_test, acc_test = test(model, data)
    wandb.log({
        "orig_model_test_loss": loss_test,
        "orig_model_test_acc": acc_test,
        "full_eval_schedule": full_schedule,
        "schedule_name": schedule_name
    })
    model_dict = model.state_dict()

    for iter in range(len(full_schedule)):
        eval_schedule = full_schedule[0:iter]
        test_model = iterativeGCN(input_dim=num_features,
                            output_dim=num_classes,
                            hidden_dim=config.hid_dim,
                            num_train_iter=iter,
                            smooth_fac=config.smooth_fac,
                            schedule=eval_schedule,
                            dropout=config.dropout)
        test_model.load_state_dict(model_dict)
        loss_test, acc_test = test(test_model, data)
        wandb.log({
            'test_loss': loss_test,
            'test_accuracy': acc_test,
            "iteration": iter+1
        })
        del test_model
    
    wandb.finish()


config = {
    'num_epochs': 200,
    'dataset_name': "Cora",
    'noise_percent': 0.7,
    'hid_dim': 32,
    'num_iter_layers': 9,
    'smooth_fac': 0.55,
    'dropout': 0.5,
    'learning_rate': 0.002,
    'weight_decay': 4e-4
} 

full_schedule = F.sigmoid(torch.Tensor(np.linspace(0.5, 3, 60))).detach().cpu().numpy()
schedule_name = "Sigmoid"
# full_schedule = np.linspace(0.7, 0.95, 90)
# schedule_name = "Linear"

run_exp(config, full_schedule, schedule_name)