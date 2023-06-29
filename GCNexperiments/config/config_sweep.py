import numpy as np

sweep_config = {
    'method': 'random'
}

metric = {
    'name': 'accuracy',
    'goal': 'maximize'
}
sweep_config['metric'] = metric

parameters_dict = {
    'num_iter_layers': {
        'values': [2, 3, 4, 5, 6, 7, 8, 9]
    },
    'learning_rate': {
        'values': np.arange(0.005, 0.02, 0.0005).tolist()
    },
    'smooth_fac': {
        'values': np.arange(0.3, 1, 0.025).tolist()
    },
    'hid_dim': {
        'value': 32
    },
    'weight_decay': {
        'values': [3e-4, 4e-4, 5e-4]
    },
    'num_epochs': {
        'value': 200
    },
    'dropout': {
        'value': 0.5
    },
    'dataset_name': {
        'value': 'Cora'
    },
    'noise_percent': {
        'value': 0.7
    }
}
sweep_config['parameters'] = parameters_dict