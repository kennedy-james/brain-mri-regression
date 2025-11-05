from enum import Enum, auto
import torch.nn as nn
from torchmetrics import R2Score

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


RUNNING_MODE = RunMode.optuna_search
configs = {
    'folds': 10,
    'random_state': 42,
    'impute_method': Imputer.knn,
    'outlier_method': OutlierDetector.pca_isoforest,
    'regression_method': Regressor.stacking,
    'optuna': {
        'load_file': 'best_params_lowering_overfit.json',
        'objective_to_run': 'xgb', # stacker or xbg
    }
}

# Add configuration for imputation
if configs['impute_method'] == Imputer.knn:
    imputation_config = {
        'knn_neighbours': 75,
        'knn_weight': 'uniform',  # possible neighbour weights for average (uniform, distance)
    }
elif configs['impute_method'] == Imputer.iterative:
    imputation_config = {
        'iterative_estimator': 'Ridge()',  # Iterative configuration
        'iterative_iter': 1,  # Iterative configuration
    }
else:
    imputation_config = {}

# Add configuration for outlier detection
if configs['outlier_method'] == OutlierDetector.pca_isoforest:
    outlier_config = {
        'pca_isoforest_contamination': 0.045, # proportion of outliers
        'pca_n_components': 2,
    }
elif configs['outlier_method'] == OutlierDetector.pca_svm:
    outlier_config = {
        'pca_n_components': 2,
        'pca_svm_nu': 0.05,    # "expected amount of outliers to discard"
        'pca_svm_gamma': 0.0003, # blurriness of internal holes within clusters
    }
elif configs['outlier_method'] == OutlierDetector.zscore:
    outlier_config = {
        'zscore_std': 1
    }
elif configs['outlier_method'] == OutlierDetector.isoforest:
    outlier_config = {
        'isoforest_contamination': 0.05,
    }
else:
    outlier_config = {}

# Add configuration for feature selection
selection_config = {
    'selection_is_enabled': True,
    'selection_thresh_var': 0.01,
    'selection_thresh_corr': 0.90,
    'selection_rf_max_feats': 300,
    'selection_percentile': 40
}

# Add configuration for regression
if configs['regression_method'] == Regressor.neural_network:
    regression_config = {
        'nn_architecture': nn.Sequential(
            nn.Linear(300, 150), nn.ReLU(),
            nn.BatchNorm1d(150),
            nn.Linear(150, 50), nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, 1), nn.ReLU()
        ),
        'nn_parameters': {
            # 'criterion': R2Score,
            # batch_size=100,
            'train_split': None
        }
    }
elif configs['regression_method'] in [Regressor.xgb, Regressor.stacking]:
    regression_config = {
            'xgb_eval_metric': 'rmse',
            'xgb_early_stopping_rounds': 20
    }
elif configs['regression_method'] == Regressor.gradient_boosting:
    regression_config = {
        'gb_n_estimators': 1000,
        'gb_learning_rate': 0.1,
        'gb_max_depth': 3,
        'gb_min_samples_split': 2
    }
else:
    regression_config = {}

# Generate final configs file from components
configs = {
    **configs,
    **imputation_config,
    **outlier_config,
    **selection_config,
    **regression_config
}
