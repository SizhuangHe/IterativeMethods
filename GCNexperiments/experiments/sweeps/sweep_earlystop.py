import numpy as np
import random
import wandb
import time

def train_one_epoch(epoch, lr, bs): 
  acc = 0.25 + ((epoch/30) +  (random.random()/10))
  loss = 0.2 + (1 - ((epoch-1)/10 +  random.random()/5))
  return acc, loss

def evaluate_one_epoch(epoch):
  acc = 0.1 + ((epoch/20) +  (random.random()/10))
  loss = 0.25 + (1 - ((epoch-1)/10 +  random.random()/6))
  return acc, loss

def main():
    run = wandb.init()

    lr  =  wandb.config.lr
    bs = wandb.config.batch_size

    for epoch in range(10):
      # Sleep a sufficient amount of time each epoch for early stopping to terminate an ongoing run 
      time.sleep(35) 
      train_acc, train_loss = train_one_epoch(epoch, lr, bs)
      val_acc, val_loss = evaluate_one_epoch(epoch)

      wandb.log({
        'train_acc': train_acc,
        'train_loss': train_loss,
        'val_acc': val_acc,
        'val_loss': val_loss
      })

# 🐝 Step 2: Define sweep config
sweep_configuration = {
    'method': 'random',
    'name': 'hyperband_sweep_test',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters':
    {
        'batch_size': {'values': [16, 32, 64]},
        'lr': {'max': 0.1, 'min': 0.0001}
     },
     "early_terminate": {
        "type": "hyperband",
        "eta": 2,
        "min_iter":2
     }
}


# run once to obtain a sweep ID
sweep_id = wandb.sweep(sweep=sweep_configuration, project='hyperband_example_project2')

# 🐝 Step 4: Call to `wandb.agent` to start a sweep
wandb.agent(sweep_id, function=main, count=50)