from enum import Enum, auto


class RunMode(Enum):
    final_evaluation = auto() # produce submission file for test data
    wandb = auto() # log to wandb
    wandb_sweep = auto() # run a wandb sweep for hyperparameter optimization
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
    gradient_boosting = auto()
    stacking = auto()


RUNNING_MODE = RunMode.current_config
configs = {
    'folds': 10,
    'random_state': 42,
    'impute_method': Imputer.knn,
    'knn_neighbours': 75,
    'knn_weight': 'uniform',  # possible neighbour weights for average (uniform, distance)
    'iterative_estimator': 'Ridge()',  # Iterative configuration
    'iterative_iter': 1,  # Iterative configuration
    'outlier_detector': {
        'method': OutlierDetector.pca_isoforest,
        'zscore_std': 1,
        'isoforest_contamination': 0.05,
        'pca_n_components': 2,
        'pca_svm_nu': 0.05,    # "expected amount of outliers to discard"
        'pca_svm_gamma': 0.0003, # blurriness of internal holes within clusters
        'pca_isoforest_contamination': 0.045, # proportion of outliers
    },
    'selection': {
        'is_enabled': True,
        'thresh_var': 0.01,
        'thresh_corr': 0.90,
        'rf_max_feats': 300,
        'percentile': 40,
    },
    'regression_method': Regressor.stacking,
    'xgboost': {
        'eval_metric': 'rmse',
        'early_stopping_rounds': 20
    },

    'gb_n_estimators': 1000,
    'gb_learning_rate': 0.1,
    'gb_max_depth': 3,
    'gb_min_samples_split': 2,

    'optuna': {
        'load_file': 'best_params_lowering_overfit.json',
        'objective_to_run': 'stacker', # stacker or xbg
    }
}
