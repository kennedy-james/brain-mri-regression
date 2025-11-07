import optuna
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
    StackingRegressor, BaggingRegressor
)
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

# Imports from your existing utils_optuna.py
from cosinecows.config import Imputer, configs, OutlierDetector, Regressor
from cosinecows.feature_selection import feature_selection
from cosinecows.imputation import imputation
from cosinecows.modeling.train import run_cv_experiment  # Used by 'objective'

# Import needed for the new objective_stacker
from cosinecows.outlier_detection import outlier_detection

# Global var for CV
N_SPLITS = int(configs.get("cv_folds", 5)) if configs.get("cv_folds") else 5


def objective(trial, x, y):
    """
    Main function for Optuna hyperparam optimization.
    This new version tunes the *robust, full-feature* tree-based pipeline.
    """

    # --- 1. Tune Pipeline Steps ---
    impute_method_name = trial.suggest_categorical(
        'impute_method', ['knn', 'iterative']
    )
    configs['impute_method'] = Imputer[impute_method_name]

    # We will only tune the full-data, robust outlier detectors
    outlier_method_name = trial.suggest_categorical(
        'outlier_method_name', ['isoforest', 'zscore']
    )
    configs['outlier_detector']['method'] = OutlierDetector[outlier_method_name]

    # --- 2. Tune Pipeline Hyperparameters (Conditional) ---
    if outlier_method_name == 'isoforest':
        # Note: We are tuning the correct contamination parameter now
        configs['outlier_detector']['isoforest_contamination'] = trial.suggest_float(
            'isoforest_contamination', low=0.01, high=0.1
        )
    elif outlier_method_name == 'zscore':
        configs['outlier_detector']['zscore_std'] = trial.suggest_float(
            'zscore_std', low=1.0, high=2.5
        )

    # --- 3. Tune Model (XGBoost) ---
    # We hard-code the regressor, which will trigger the PassthroughSelector
    configs['regression_method'] = Regressor.xgb

    # Tune XGBoost parameters for a HIGH-DIMENSIONAL (832 features) dataset
    configs['regression_params'] = {
        'random_state': configs["random_state"],
        'n_estimators': trial.suggest_int('n_estimators', low=100, high=500),
        'max_depth': trial.suggest_int('max_depth', low=3, high=8),
        'min_child_weight': trial.suggest_int('min_child_weight', low=10, high=25),
        'gamma': trial.suggest_float('gamma', low=0.5, high=3.0),
        'subsample': trial.suggest_float('subsample', low=0.6, high=1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', low=0.3, high=0.7),
        'reg_alpha': trial.suggest_float('reg_alpha', low=0.1, high=2.5),
        'reg_lambda': trial.suggest_float('reg_lambda', low=2.0, high=6.0),  # Higher L2 for high-D
        'learning_rate': trial.suggest_float('learning_rate', low=0.01, high=0.1, log=True),
        'verbosity': 0
    }

    # --- 4. Run the Experiment ---
    configs['folds'] = 3  # use fewer folds for faster tuning.

    try:
        cv_df = run_cv_experiment(x, y)
        mean_val_score = cv_df['validation_score'].mean()
    except Exception as e:
        print(f"--- ❌ TRIAL FAILED: {e} ---")
        return -1.0  # Return a very bad score

    return mean_val_score


# --- Helper functions for new objective_stacker ---

def _make_xgb_like(trial, name_prefix="xgb"):
    """Return a GradientBoostingRegressor approximating an XGBoost-like learner (safe fallback)."""
    lr = trial.suggest_float(f"{name_prefix}_learning_rate", 0.01, 0.3, log=True)
    n_estimators = trial.suggest_int(f"{name_prefix}_n_estimators", 50, 1000, step=50)
    max_depth = trial.suggest_int(f"{name_prefix}_max_depth", 2, 12)
    subsample = trial.suggest_float(f"{name_prefix}_subsample", 0.5, 1.0)
    max_features = trial.suggest_categorical(f"{name_prefix}_max_features", ["auto", "sqrt", "log2", None])
    return GradientBoostingRegressor(
        learning_rate=lr,
        n_estimators=n_estimators,
        max_depth=max_depth,
        subsample=subsample,
        max_features=None if max_features == "auto" else max_features,
        random_state=configs["random_state"],
    )


def _make_rf(trial, name_prefix="rf"):
    n_estimators = trial.suggest_int(f"{name_prefix}_n_estimators", 50, 1000, step=50)
    max_depth = trial.suggest_int(f"{name_prefix}_max_depth", 3, 50)
    max_features = trial.suggest_categorical(f"{name_prefix}_max_features", ["sqrt", "log2", 0.3, 0.5, None])
    min_samples_split = trial.suggest_int(f"{name_prefix}_min_samples_split", 2, 10)
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=None if max_features is None else max_features,
        min_samples_split=min_samples_split,
        n_jobs=-1,
        random_state=configs["random_state"],
    )


def _make_et(trial, name_prefix="et"):
    n_estimators = trial.suggest_int(f"{name_prefix}_n_estimators", 50, 800, step=50)
    max_depth = trial.suggest_int(f"{name_prefix}_max_depth", 3, 50)
    max_features = trial.suggest_categorical(f"{name_prefix}_max_features", ["sqrt", "log2", 0.3, 0.5, None])
    return ExtraTreesRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=None if max_features is None else max_features,
        n_jobs=-1,
        random_state=configs["random_state"],
    )


def _make_svr(trial, name_prefix="svr"):
    C = trial.suggest_float(f"{name_prefix}_C", 0.1, 100.0, log=True)
    eps = trial.suggest_float(f"{name_prefix}_epsilon", 1e-4, 1.0, log=True)
    kernel = trial.suggest_categorical(f"{name_prefix}_kernel", ["rbf", "poly", "linear"])
    if kernel == "poly":
        degree = trial.suggest_int(f"{name_prefix}_degree", 2, 5)
    else:
        degree = 3
    return SVR(C=C, epsilon=eps, kernel=kernel, degree=degree)


def _make_meta(trial):
    meta_choice = trial.suggest_categorical("meta_estimator", ["ridge", "lasso", "svr", "rf"])

    # 1. Create the base meta-model
    if meta_choice == "ridge":
        alpha = trial.suggest_float("meta_ridge_alpha", 1e-4, 10.0, log=True)
        base_meta_model = Ridge(alpha=alpha, random_state=configs["random_state"])
    elif meta_choice == "lasso":
        alpha = trial.suggest_float("meta_lasso_alpha", 1e-6, 1.0, log=True)
        base_meta_model = Lasso(alpha=alpha, random_state=configs['random_state'], max_iter=5000)
    elif meta_choice == "svr":
        # SVR as a meta-model needs scaled inputs
        base_meta_model = make_pipeline(
            StandardScaler(),
            _make_svr(trial, name_prefix="meta_svr")
        )
    else:  # 'rf'
        base_meta_model = _make_rf(trial, name_prefix="meta_rf")

    # Optionally wrap the meta-model in a BaggingRegressor
    use_meta_bagging = trial.suggest_categorical("use_meta_bagging", [True, False])

    if use_meta_bagging:
        n_estimators = trial.suggest_int("meta_bagging_n", 5, 20)
        return BaggingRegressor(
            base_meta_model,
            n_estimators=n_estimators,
            n_jobs=-1,
            random_state=configs["random_state"]
        )
    else:
        # Just return the unwrapped model
        return base_meta_model


def create_stacking_from_trial(trial):
    """Build a list of base estimators and meta estimator from trial parameters."""
    # select which base learners to include
    use_xgb = trial.suggest_categorical("use_xgb", [True, False])
    use_rf = trial.suggest_categorical("use_rf", [True, False])
    use_et = trial.suggest_categorical("use_et", [True, False])
    use_svr = trial.suggest_categorical("use_svr", [True, False])

    estimators = []
    idx = 0
    if use_xgb:
        idx += 1
        estimators.append((f"xgb_{idx}", _make_xgb_like(trial, name_prefix=f"xgb_{idx}")))
    if use_rf:
        idx += 1
        estimators.append((f"rf_{idx}", _make_rf(trial, name_prefix=f"rf_{idx}")))
    if use_et:
        idx += 1
        estimators.append((f"et_{idx}", _make_et(trial, name_prefix=f"et_{idx}")))
    if use_svr:
        idx += 1
        # SVR benefits from scaling -> wrap in pipeline
        estimators.append((f"svr_{idx}", make_pipeline(StandardScaler(), _make_svr(trial, name_prefix=f"svr_{idx}"))))

    # fallback: ensure at least one estimator
    if not estimators:
        estimators.append(
            ("rf_default", RandomForestRegressor(n_estimators=200, random_state=configs['random_state'], n_jobs=-1)))

    meta = _make_meta(trial)

    passthrough = trial.suggest_categorical("passthrough", [True, False])
    n_jobs = -1

    stacking = StackingRegressor(
        estimators=estimators,
        final_estimator=meta,
        passthrough=passthrough,
        n_jobs=n_jobs,
        cv=3,  # internal stacking CV; keep small for speed (tunable if desired)
    )
    return stacking


def objective_stacker(trial: optuna.Trial, x, y):
    """
    Optuna objective: build stacking regressor from trial and evaluate CV R² mean.
    This version replicates the full train pipeline (impute -> outlier remove -> train).
    """

    # 1. Build the *model* part of the pipeline (scaler + stacker)
    model = create_stacking_from_trial(trial)

    pipeline_steps = []
    use_scaler = trial.suggest_categorical("use_scaler_before_cv", [True, False])
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))
    pipeline_steps.append(("stacker", model))

    # This pipeline now only contains the model and its (optional) scaler
    model_pipeline = Pipeline(steps=pipeline_steps)

    # 2. Manually run cross-validation to control for outlier removal
    # NOTE: We use N_SPLITS=3 for speed during optimization
    cv = KFold(n_splits=3, shuffle=True, random_state=configs['random_state'])
    scores = []

    try:
        for train_index, val_index in cv.split(x, y):
            # Split data
            x_train, x_val = x[train_index], x[val_index]
            y_train, y_val = y[train_index], y[val_index]

            # 3. Impute Data (fit on train, transform both)
            imputer = imputation(x_train, i=None)
            x_train_imp = imputer.fit_transform(x_train)
            x_val_imp = imputer.transform(x_val)  # Use this for validation

            # 4. Tune and apply Outlier Detector (fit on train, apply to train ONLY)
            outlier_method_name = trial.suggest_categorical(
                'outlier_method', ['pca_isoforest', 'isoforest']
            )

            if outlier_method_name == 'none':
                x_train_clean = x_train_imp
                y_train_clean = y_train
                # We also need the detector to apply to the validation set
                detector = None
            else:
                configs['outlier_detector']['method'] = OutlierDetector[outlier_method_name]
                if outlier_method_name == 'pca_isoforest':
                    configs['outlier_detector']['pca_isoforest_contamination'] = trial.suggest_float(
                        'pca_isoforest_contamination', 0.01, 0.05)
                    configs['outlier_detector']['pca_n_components'] = trial.suggest_int('pca_n_components', 5, 20)
                elif outlier_method_name == 'isoforest':
                    configs['outlier_detector']['isoforest_contamination'] = trial.suggest_float(
                        'isoforest_contamination', 0.01, 0.1)
                elif outlier_method_name == 'zscore':
                    configs['outlier_detector']['zscore_std'] = trial.suggest_float('zscore_std', 1.0, 2.5)

                detector = outlier_detection(x_train_imp, y_train)
                train_mask = detector(x_train_imp)  # Get mask from train data
                x_train_clean = x_train_imp[train_mask]  # Apply mask to features
                y_train_clean = y_train[train_mask]  # Apply mask to labels

            selection = feature_selection(
                x_train_clean,
                y_train_clean,
                thresh_var=configs['selection']['thresh_var'],
                thresh_corr=configs['selection']['thresh_corr'],
                # rf_max_feats=configs['selection']['rf_max_feats'],
                percentile=configs['selection']['percentile']
            )
            x_proc = selection.transform(x_train_clean)

            # 5. Fit the model pipeline on the *cleaned* training data
            model_pipeline.fit(x_proc, y_train_clean)

            # --- START: MODIFIED STEP 6 ---
            # 6. Score the model on the *cleaned* validation data (as requested)

            if detector is not None:
                val_mask = detector(x_val_imp)
                x_val_clean = x_val_imp[val_mask]
                y_val_clean = y_val[val_mask]
            else:
                # If outlier_method was 'none', don't clean validation data
                x_val_clean = x_val_imp
                y_val_clean = y_val

            x_val_proc = selection.transform(x_val_clean)
            y_pred = model_pipeline.predict(x_val_proc)

            # Score against the *cleaned* validation labels
            scores.append(r2_score(y_val_clean, y_pred))
            # --- END: MODIFIED STEP 6 ---

        # 7. Report the mean score
        mean_score = float(np.mean(scores))
        trial.report(mean_score, 0)
        return mean_score

    except Exception as e:  # Catch errors during the CV process
        print(f"--- ❌ TRIAL FAILED: {e} ---")
        return -1.0  # Return a bad score to prune the trial
