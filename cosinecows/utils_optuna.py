from cosinecows.config import Imputer, configs, OutlierDetector, Regressor
from cosinecows.modeling.train import run_cv_experiment
from sklearn.gaussian_process.kernels import RationalQuadratic
#catboost


def objective(trial, x, y):
    """
    Main function for Optuna hyperparam optimization.
    This new version tunes the *robust, full-feature* tree-based pipeline.
    """

    # --- 1. Tune Pipeline Steps ---
    #impute_method_name = trial.suggest_categorical('impute_method', ['knn', 'iterative'])
    impute_method_name = 'knn'  # Fix to KNN for now
    configs['impute_method'] = Imputer[impute_method_name]

    # We will only tune the full-data, robust outlier detectors
    #outlier_method_name = trial.suggest_categorical('outlier_method_name', ['pca_isoforest', 'zscore'])
    outlier_method_name = 'pca_isoforest'  # Fix to PCA + IsoForest for now
    configs['outlier_method'] = OutlierDetector.pca_isoforest

    # --- 2. Tune Pipeline Hyperparameters (Conditional) ---
    if outlier_method_name == 'isoforest':
        # Note: We are tuning the correct contamination parameter now
        configs['isoforest_contamination'] = trial.suggest_float(
            'isoforest_contamination', low=0.01, high=0.1
        )
    elif outlier_method_name == 'zscore':
        configs['zscore_std'] = trial.suggest_float(
            'zscore_std', low=1.0, high=2.5
        )

    

    # --- 3. Tune Model (XGBoost) ---
    # We hard-code the regressor, which will trigger the PassthroughSelector

    # Tune XGBoost parameters for a HIGH-DIMENSIONAL (832 features) dataset
    if configs['regression_method'] == Regressor.xgb:
        configs['regression_params'] = {
            'random_state': configs["random_state"],
            'n_estimators': trial.suggest_int('n_estimators', low=400, high=2500),
            'max_depth': trial.suggest_int('max_depth', low=4, high=9),
            'min_child_weight': trial.suggest_int('min_child_weight', low=10, high=25),
            'gamma': trial.suggest_float('gamma', low=0.5, high=3.0),
            'subsample': trial.suggest_float('subsample', low=0.65, high=1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', low=0.3, high=1.0), 
            'reg_alpha': trial.suggest_float('reg_alpha', low=0.1, high=2.5),
            #'reg_alpha': 2,
            'reg_lambda': trial.suggest_float('reg_lambda', low=2.0, high=6.0),  # Higher L2 for high-D
            #'reg_lambda': 4.0,
            'learning_rate': trial.suggest_float('learning_rate', low=0.01, high=0.1, log=True),
            #'learning_rate': 0.05,
            'verbosity': 0
        }
    if configs['regression_method'] == Regressor.gaussian_process:
        #configs['selection_percentile'] = trial.suggest_int('selection_percentile', 15, 50)
        #configs['selection_rf_max_feats'] = trial.suggest_int('selection_rf_max_feats', 25, 125)
        #configs['regression_params'] = {
        #    'random_state': configs["random_state"],
        #    'length_scale': trial.suggest_float('length_scale', low=4, high=10),
        #    'alpha': trial.suggest_float('alpha', low=0.4, high=0.8),
        #    'gp_alpha': trial.suggest_float('gp_alpha', low=1.0e-10, high=1.0e-7, log=True),
        #}
        #{'selection_percentile': 32, 'selection_rf_max_feats': 44, 'length_scale': 6.124209435262154, 'alpha': 0.669737299146556, 'gp_alpha': 2.965074241784881e-09}
        configs['selection_percentile'] = 32
        configs['selection_rf_max_feats'] = 44
        configs['regression_params']['length_scale'] = 6.124209435262154
        configs['regression_params']['alpha'] = 0.669737299146556
        configs['regression_params']['gp_alpha'] = 2.965074241784881e-09
    if configs['regression_method'] == Regressor.neural_network:
        configs['selection_percentile'] = trial.suggest_int('selection_percentile', 15, 75)
        configs['selection_rf_max_feats'] = trial.suggest_int('selection_rf_max_feats', 25, 200)

        configs['nn_optimizer'] = trial.suggest_categorical('optimizer', ['opt.Adamax', 'opt.RAdam'])
        configs['nn_loss'] = trial.suggest_categorical('loss', ['nn.MSELoss', 'nn.HuberLoss', 'nn.SmoothL1Loss'])
        configs['nn_parameters']['lr'] = trial.suggest_float('learning_rate', low=0.0001, high=10, log=True)
        configs['nn_parameters']['batch_size'] = trial.suggest_int('batch_size', low=35, high=275)
        configs['nn_parameters']['max_epochs'] = trial.suggest_int('max_epochs', low=3, high=15)
        configs['nn_depth'] = trial.suggest_int('depth', low=1, high=4)

        configs['nn_width'] = [configs['selection_rf_max_feats']]
        configs['nn_dropout'] = []
        configs['nn_activation'] = []
        for layer in range(configs['nn_depth']):
            configs['nn_width'].append(trial.suggest_int(f'width_layer{layer + 1}', low=2, high=configs['nn_width'][layer]))
        configs['nn_width'].append(1)

        for layer in range(configs['nn_depth'] + 1):
            configs['nn_dropout'].append(trial.suggest_float(f'dropout_layer{layer + 1}', low=0, high=0.65))
            configs['nn_activation'].append(trial.suggest_categorical(f'activation_layer{layer + 1}', [
                'nn.ReLU', 'nn.SiLU', 'nn.Tanh', 'nn.Hardswish', 'nn.ELU', 'nn.PReLU', 'nn.SELU',
                'nn.Hardtanh', 'nn.Hardshrink', 'nn.Tanhshrink', 'nn.Softsign', 'nn.Softshrink',
                'nn.Mish', 'nn.Sigmoid', 'nn.GELU', 'nn.RReLU', 'nn.ReLU6', 'nn.LogSigmoid',
                'nn.LeakyReLU']))
    if configs['regression_method'] == Regressor.tab_net:
        configs['selection_percentile'] = trial.suggest_int('selection_percentile', 15, 75)
        configs['selection_rf_max_feats'] = trial.suggest_int('selection_rf_max_feats', 25, 200)

        configs['optimizer_fn'] = trial.suggest_categorical('optimizer_fn', [
            'opt.Adadelta', 'opt.Adafactor', 'opt.Adagrad', 'opt.Adam', 'opt.AdamW', 'opt.Adamax', 'opt.ASGD',
            'opt.NAdam', 'opt.RAdam', 'opt.RMSprop', 'opt.Rprop'])

        configs['tab_parameters']['n_d'] = trial.suggest_int('n_d', low=8, high=64)
        configs['tab_parameters']['n_a'] = configs['tab_parameters']['n_d']
        # configs['n_a'] = trial.suggest_int('n_a', low=8, high=64)
        configs['tab_parameters']['n_steps'] = trial.suggest_int('n_steps', low=3, high=10)
        configs['tab_parameters']['gamma'] = trial.suggest_float('gamma', low=1.0, high=2.0)
        configs['tab_parameters']['n_independent'] = trial.suggest_int('n_independent', low=1, high=5)
        configs['tab_parameters']['n_shared'] = trial.suggest_int('n_shared', low=1, high=5)
        configs['tab_parameters']['momentum'] = trial.suggest_float('momentum', low=0.01, high=0.4)

        configs['tab_fitting']['max_epochs'] = trial.suggest_int('max_epochs', low=50, high=250)
        configs['tab_fitting']['virtual_batch_size'] = trial.suggest_int('virtual_batch_size', low=64, high=256)
        configs['tab_fitting']['patience'] = trial.suggest_int('patience', low=0, high=15)
        configs['tab_fitting']['warm_start'] = trial.suggest_int('warm_start', low=0, high=1)
    if configs['regression_method'] == Regressor.extra_trees:
        configs['xtrees_parameters']['n_estimators'] = trial.suggest_int('n_estimators', low=50, high=250)
        configs['xtrees_parameters']['max_depth'] = trial.suggest_int('max_depth', low=5, high=40)
        configs['xtrees_parameters']['min_samples_split'] = trial.suggest_int('min_samples_split', low=2, high=10)
        configs['xtrees_parameters']['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', low=1, high=10)
        configs['xtrees_parameters']['bootstrap'] = trial.suggest_categorical('bootstrap', [True, False])
        configs['xtrees_parameters']['max_features'] = trial.suggest_float('max_features', low=0.1, high=1.0)
        configs['xtrees_parameters']['ccp_alpha'] = trial.suggest_float('ccp_alpha', low=0.0, high=0.5)


        

    # if configs['regression_method'] == Regressor.catboost:
    #     configs['regression_params'] = {
    #         'random_state': configs["random_state"],
    #         'iterations': trial.suggest_int('iterations', low=100, high=1000),
    #         'learning_rate': trial.suggest_float('learning_rate', low=0.01, high=0.3, log=True),
    #         'depth': trial.suggest_int('depth', low=4, high=10),
    #         'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', low=1.0, high=10.0),
    #     }

    # --- 4. Run the Experiment ---
    configs['folds'] = 4  # use fewer folds for faster tuning.

    try:
        cv_df = run_cv_experiment(x, y)
        mean_val_score = cv_df['validation_score'].mean()
    except Exception as e:
        print(f"--- ❌ TRIAL FAILED: {e} ---")
        return -1.0  # Return a very bad score

    return mean_val_score


def objective_stacker(trial, x, y):
    """
    Optuna objective function for tuning the STACKING PIPELINE.
    This tunes the pre-processing steps, not the models themselves.
    """
    print(f"\n--- 🚀 Optuna Trial {trial.number} (Stacker Pipeline) ---")

    # --- 1. Tune Pipeline Hyperparameters ---

    # Tune Feature Selection Percentile
    # This is the most important parameter to tune.
    percentile = trial.suggest_int('selection_percentile', 20, 60)
    configs['selection_percentile'] = percentile

    # Tune Outlier Detector (the one you are currently using)
    configs['outlier_method'] = OutlierDetector.pca_isoforest

    contamination = trial.suggest_float('pca_isoforest_contamination', 0.01, 0.05)
    configs['pca_isoforest_contamination'] = contamination

    n_components = trial.suggest_int('pca_n_components', 5, 20)
    configs['pca_n_components'] = n_components

    # --- 2. Set Model ---
    # We are explicitly tuning the pipeline FOR the stacking regressor
    configs['regression_method'] = Regressor.stacking

    # --- 3. Run the Experiment ---
    configs['folds'] = 3  # Use fewer folds for faster tuning

    try:
        cv_df = run_cv_experiment(x, y)
        mean_val_score = cv_df['validation_score'].mean()
        print(f"--- ✅ Trial {trial.number} Result: {mean_val_score:.4f} ---")
    except Exception as e:
        print(f"--- ❌ TRIAL FAILED: {e} ---")
        return -1.0  # Return a very bad score

    return mean_val_score

