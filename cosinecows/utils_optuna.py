from cosinecows.config import Imputer, configs, OutlierDetector, Regressor
from cosinecows.modeling.train import run_cv_experiment
from sklearn.gaussian_process.kernels import RationalQuadratic


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
    configs['outlier_detector']['method'] = OutlierDetector[outlier_method_name]

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
    configs['regression_method'] = Regressor.xgb # Regressor.gaussian_process

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
        configs['regression_params'] = {
            'random_state': configs["random_state"],
            'length_scale': trial.suggest_float('length_scale', low=4, high=10),
            'alpha': trial.suggest_float('alpha', low=1.0e-10, high=1.0e-7, log=True), 

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

