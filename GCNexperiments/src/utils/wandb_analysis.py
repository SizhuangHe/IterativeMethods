import pandas as pd
import matplotlib.pyplot as plt
import wandb
import numpy as np
import seaborn as sns

def get_sweep_info(sweep_id_list):
    api = wandb.Api()
    dict_list = []
    for sweep_id in sweep_id_list:
        sweep_path = "sizhuang/IterativeMethods/" + sweep_id
        sweep = api.sweep(sweep_path)
        sweep_runs = sweep.runs
        num_runs = len(sweep_runs)
        sweep_dict = {}
        for k,v in sweep.config.copy()["parameters"].items():
            for _,vv in v.items():
                sweep_dict.update({k: vv})   
        sweep_dict.update({"Number of runs": num_runs})  
        sweep_dict.update({"Sweep id": sweep_id})
        dict_list.append(sweep_dict) 
    api.flush()
    df = pd.DataFrame(dict_list)
    return df

def get_raw_sweep_runs(sweep_id):
    '''
    Given sweep_id, this function returns a pandas.DataFrame that summarizes information of each runs of this sweep
    '''
    api = wandb.Api()
    sweep_path = "sizhuang/IterativeMethods/" + sweep_id
    sweep = api.sweep(sweep_path)
    sweep_runs = sweep.runs
    summary_list, config_list, name_list = [], [], []

    for run in sweep_runs:
        summary_list.append(run.summary)
        config_list.append(run.config)
        name_list.append(run.name)
        runs_df = pd.DataFrame({
            "summary": summary_list,
            "config": config_list,
            "name": name_list
        })

    api.flush()
    return runs_df

def clean_keys(dict):
    dic = dict.copy()
    for k,_ in dict.items():
        if k.startswith('gradient') or k.startswith('parameters') or k.startswith('_') or (k =='dropout') or (k == 'hid_dim') or (k == 'num_epochs'):
            dic.pop(k)
    return dic

def get_clean_sweep_runs(sweep_id, model_name):
    '''
    Given sweep_id, this function returns a pandas.DataFrame that summarizes information of each runs of this sweep
    '''
    api = wandb.Api()
    sweep_path = "sizhuang/IterativeMethods/" + sweep_id
    sweep = api.sweep(sweep_path)
    sweep_runs = sweep.runs
    
    run_list = []
    for run in sweep_runs:
        run_dict = {}
        run_dict.update(clean_keys(run.summary._json_dict))
        run_dict.update(clean_keys(run.config))
        run_dict.update({"run_name": run.name})
        run_dict.update({"model_name": model_name})
        run_list.append(run_dict)

    
    runs_df = pd.DataFrame(run_list)

    api.flush()
    return runs_df


def get_metrics(sweep_id):
    '''
    Get a list of test accuracies and a list of number of iterations/layers from a sweep
    '''
    runs_df = get_clean_sweep_runs(sweep_id)
    num_runs = len(runs_df)

    test_accuracies = runs_df['test_accuracy'].tolist()
    num_iters = runs_df['num_iter_layers'].tolist()
    assert len(test_accuracies)  == num_runs
    assert len(num_iters) == num_runs

    return test_accuracies, num_iters
    