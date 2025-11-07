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
        if not configs['ensemble_search']:
            configs['selection_percentile'] = trial.suggest_int('selection_percentile', 15, 50)
            configs['selection_rf_max_feats'] = trial.suggest_int('selection_rf_max_feats', 40, 300)
            # 5 neightbours or 75 neighbours for KNN imputer
            configs['knn_neighbours'] = trial.suggest_categorical('knn_neighbours', [3, 5, 7, 10])


            configs['regression_params'] = { 
                'random_state': configs["random_state"],
                #'n_estimators': trial.suggest_int('n_estimators', low=400, high=2500),
                'n_estimators': 10000,
                'max_depth': trial.suggest_int('max_depth', low=6, high=15),
                'min_child_weight': trial.suggest_int('min_child_weight', low=10, high=25),
                'gamma': trial.suggest_float('gamma', low=1.0, high=2.5),
                'subsample': trial.suggest_float('subsample', low=0.4, high=1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', low=0.3, high=1.0), 
                'reg_alpha': trial.suggest_float('reg_alpha', low=0.5, high=2.0),
                #'reg_alpha': 2,
                'reg_lambda': trial.suggest_float('reg_lambda', low=1.5, high=5.0),  # Higher L2 for high-D
                #'reg_lambda': 4.0,
                'learning_rate': trial.suggest_float('learning_rate', low=0.0006, high=0.004, log=True),
                #'learning_rate': 0.05,
                'early_stopping_rounds': 100,
                'verbosity': 0
            }
        else:
            #load from best params.json
            #best_params_xgb_7_10_second.json

            with open('best_params_xgb_7_10_second.json', 'r') as f:
                best_params = json.load(f)

            
            # Assign top-level selections
            configs['selection_percentile'] = best_params['selection_percentile']
            configs['selection_rf_max_feats'] = best_params['selection_rf_max_feats']
            configs['knn_neighbours'] = best_params['knn_neighbours']

            # Assign regression parameters
            configs['regression_params'] = {
                'random_state': configs['random_state'],
                'n_estimators': 10000,  # keep fixed
                'max_depth': best_params['max_depth'],
                'min_child_weight': best_params['min_child_weight'],
                'gamma': best_params['gamma'],
                'subsample': best_params['subsample'],
                'colsample_bytree': best_params['colsample_bytree'],
                'reg_alpha': best_params['reg_alpha'],
                'reg_lambda': best_params['reg_lambda'],
                'learning_rate': best_params['learning_rate'],
                'early_stopping_rounds': 100,
                'verbosity': 0
            }

    if configs['regression_method'] == Regressor.gaussian_process:
        configs['selection_percentile'] = trial.suggest_int('selection_percentile', 20, 50)
        configs['selection_rf_max_feats'] = trial.suggest_int('selection_rf_max_feats', 30, 50)
        
        configs['regression_params'] = {
            'random_state': configs["random_state"],
            'length_scale': trial.suggest_float('length_scale', low=4, high=10),
            'alpha': trial.suggest_float('alpha', low=0.4, high=0.8),
            'noise_level': trial.suggest_float('noise_level', low=0.02, high=0.3, log=True),
            'gp_alpha': trial.suggest_float('gp_alpha', low=1.0e-10, high=1.0e-7, log=True),
            'n_restarts_optimizer': trial.suggest_int('n_restarts_optimizer', low=2, high=10),
        }
        #{'selection_percentile': 32, 'selection_rf_max_feats': 44, 'length_scale': 6.124209435262154, 'alpha': 0.669737299146556, 'gp_alpha': 2.965074241784881e-09}
        #configs['selection_percentile'] = 32
        #configs['selection_rf_max_feats'] = 44
        #configs['regression_params']['length_scale'] = 6.124209435262154
        #configs['regression_params']['alpha'] = 0.669737299146556
        #configs['regression_params']['gp_alpha'] = 2.965074241784881e-09

    if configs['regression_method'] == Regressor.catboost:
        configs['regression_params'] = {
            'random_state': configs["random_state"],
            'iterations': trial.suggest_int('iterations', low=100, high=1000),
            'learning_rate': trial.suggest_float('learning_rate', low=0.01, high=0.3, log=True),
            'depth': trial.suggest_int('depth', low=4, high=10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', low=1.0, high=10.0),
        }

    # --- 4. Run the Experiment ---
    configs['folds'] = 4  # use fewer folds for faster tuning.

    try:
        cv_df = run_cv_experiment(x, y)
        mean_val_score = cv_df['validation_score'].mean()
    except Exception as e:
        print(f"--- ❌ TRIAL FAILED: {e} ---")
        return -1.0  # Return a very bad score

    return mean_val_score

def input_best_


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

