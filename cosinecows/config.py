from enum import Enum, auto
import torch.nn as nn
from torchmetrics import R2Score
import torch.optim as opt

class RunMode(Enum):
    final_evaluation = auto() # produce submission file for test data
    wandb = auto() # log to wandb
    grid = auto()   # run all combinations of models and outlier detectors locally
    current_config = auto() # run single CV with current config
    optuna_search = auto()
    optuna_config = auto()


class Imputer(Enum):
    mean = auto()
    median = auto()
    most_frequent = auto()
    knn = auto()
    iterative = auto()


class OutlierDetector(Enum):
    zscore = auto()
    knn = auto()
    isoforest = auto()
    svm = auto()
    pca_svm = auto()
    pca_isoforest = auto()


class Regressor(Enum):
    xgb = auto()
    extra_trees = auto()
    ridge = auto()
    random_forest_regressor = auto()
    gradient_boosting = auto() # TODO: Graident boosting doesn't actually seem to be implemented?
    stacking = auto()
    neural_network = auto()
    gaussian_process = auto()
    tab_net = auto()
    svr = auto()


RUNNING_MODE = RunMode.current_config
configs = {
    'folds': 5,
    'random_state': 42,
    'impute_method': Imputer.knn,
    'outlier_method': OutlierDetector.pca_isoforest,
    'regression_method': Regressor.stacking,
    'optuna': {
        'load_file': 'best_params_xgb.json',
        'objective_to_run': 'stacker', # stacker or xbg
    }
}

# Add configuration for imputation
match configs['impute_method']:
    case Imputer.knn:
        imputation_config = {
            'knn_neighbours': 40,
            'knn_weight': 'uniform',  # possible neighbour weights for average (uniform, distance)
        }
    case Imputer.iterative:
        imputation_config = {
            'iterative_estimator': 'Ridge()',  # Iterative configuration
            'iterative_iter': 1,  # Iterative configuration
        }
    case _:
        imputation_config = {}

match configs['outlier_method']:
    case OutlierDetector.pca_isoforest:
        outlier_config = {
            'pca_isoforest_contamination': 0.045, # proportion of outliers
            'pca_n_components': 2,
        }
    case OutlierDetector.pca_svm:
        outlier_config = {
            'pca_n_components': 2,
            'pca_svm_nu': 0.05,     # expected amount of outliers to discard
            'pca_svm_gamma': 0.0003, # blurriness of internal holes within clusters
        }
    case OutlierDetector.zscore:
        outlier_config = {
            'zscore_std': 1,
        }
    case OutlierDetector.isoforest:
        outlier_config = {
            'isoforest_contamination': 0.05,
        }
    case _:
        outlier_config = {}


# Add configuration for regression
match configs['regression_method']:
    case Regressor.neural_network:
        nn_definition = {}

        regression_config = {
            # Configuration to define architecture of NN
            'nn_depth': 2,
            'nn_dropout': [0.2, 0.2, 0.2],
            'nn_width': [300, 150, 150, 1],
            'nn_activation': [nn.ReLU, nn.ReLU, nn.ReLU],

            # Configuration for NN training
            'nn_optimizer': 'opt.Adamax', # Note: RAdam also good # Outside param dict otherwise optuna fails
            'nn_loss': 'nn.MSELoss',
            'nn_parameters': {
                'batch_size': 128,
                'train_split': None,
                'lr': 0.01,
                'max_epochs': 10
            }
        }
    case Regressor.xgb | Regressor.stacking:
        regression_config = {
            'xgb_eval_metric': 'rmse',
            'xgb_early_stopping_rounds': 400
        }
    case Regressor.gradient_boosting:
        regression_config = {
            'gb_n_estimators': 1000,
            'gb_learning_rate': 0.1,
            'gb_max_depth': 3,
            'gb_min_samples_split': 2
        }
    case Regressor.gaussian_process:
        regression_config = {
            # kernel rational quadratic
            'gp_kernel_length_scale': 6.124209435262154,
            'gp_kernel_alpha': 0.669737299146556,
            'gp_alpha': 2.965074241784881e-09
        }
    case Regressor.tab_net:
        regression_config = {
            'optimizer_fn': 'opt.Adam', 
            'tab_parameters': {
                'n_d': 8,
                'n_a': 8,
                'n_steps': 3,
                'gamma': 1.3,
                'n_independent': 2,
                'n_shared': 2,
                'momentum': 0.02,
            },
            'tab_fitting': {
                'max_epochs': 200,
                'drop_last': False, # Otherwise breaks, batch_size > dataset
                'virtual_batch_size': 128,
                'patience': 10,
                'warm_start': False
            }
        }
    case Regressor.extra_trees:
        regression_config = {
            'xtrees_parameters': {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'bootstrap': False,
                'max_features': 1.0,
                'ccp_alpha': 0.0
            }
        }
    case Regressor.svr:
        regression_config = {
            'svr_kernel': 'linear',
            'svr_C': 86.418,
            'svr_epsilon': 0.11,
            'svr_gamma': 'scale'
        }
    case _:
        regression_config = {}

selection_config = {
    'selection_is_enabled': True,
    'selection_thresh_var': 0.01,
    'selection_thresh_corr': 0.90,
    'selection_rf_max_feats': 44,
    'selection_percentile': 32,
    'selection_k_best': 194,
}


# generate final configs file from components
configs = {
    **configs,
    **imputation_config,
    **outlier_config,
    **selection_config,
    **regression_config
}
