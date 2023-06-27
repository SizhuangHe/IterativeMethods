import pandas as pd
import matplotlib.pyplot as plt
import wandb
import numpy as np
import seaborn as sns

def get_metrics(sweep_id):
    '''
    Get a list of test accuracies and a list of number of iterations/layers from a sweep
    '''

    api = wandb.Api()
    sweep_path = "sizhuang/IterativeMethods/" + sweep_id
    sweep = api.sweep(sweep_path)
    sweep_runs = sweep.runs
    num_runs = len(sweep_runs)

    summary_list, config_list, name_list = [], [], []
    for run in sweep_runs: 
    # .summary contains the output keys/values 
    #  for metrics such as accuracy.
    #  We call ._json_dict to omit large files 
        summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
        config_list.append(
        {k: v for k,v in run.config.items()
         if not k.startswith('_')})

    # .name is the human-readable name of the run.
        name_list.append(run.name)
        runs_df = pd.DataFrame({
            "summary": summary_list,
            "config": config_list,
            "name": name_list
        })

    test_accuracies = [item['test_accuracy'] for item in runs_df.summary]
    num_iters = [item['num_iter_layers'] for item in runs_df.config]
    assert len(test_accuracies)  == num_runs
    assert len(num_iters) == num_runs

    api.flush()
    return test_accuracies, num_iters
    