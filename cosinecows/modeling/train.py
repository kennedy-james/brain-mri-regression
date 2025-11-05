"""
Train models.
"""
import pandas as pd
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
from skorch import NeuralNetRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor


from cosinecows.config import configs, Regressor
from cosinecows.feature_selection import PassthroughSelector, feature_selection
from cosinecows.imputation import imputation
from cosinecows.outlier_detection import outlier_detection


def fit(X, y):
    """Training of the model

    Parameters
    ----------
    X: Training data
    y: Training labels

    Returns
    ----------
    model: Final model for prediction
    """
    model_name = configs["regression_method"]
    print(f"Fitting model: {model_name}")

    if model_name is Regressor.xgb:
        # check if optuna provided set of tuned params
        if 'regression_params' in configs:
            print("Using tuned XGBoost parameters from Optuna...")
            model = XGBRegressor(**configs['regression_params'])
        else:
            model = XGBRegressor(
                random_state=configs["random_state"],
                n_estimators=300,
                max_depth=5,  # shallower trees
                min_child_weight=15,  # require more samples per leaf
                gamma=1.0,  # stronger split penalty
                subsample=0.75,
                colsample_bytree=0.4,  # fewer features per tree
                reg_alpha=0.5,
                reg_lambda=3.0,
                learning_rate=0.03,
                eval_metric=configs['xgb_eval_metric'],
                early_stopping_rounds=configs['xgb_early_stopping_rounds'],
                verbosity=0
            )
    elif model_name is Regressor.gaussian_process:

        if 'regression_params' in configs:
            print("Using tuned GPR parameters from Optuna...")
            kernel = RationalQuadratic(length_scale=configs['regression_config']['length_scale'],
                                       alpha=configs['regression_config']['alpha'])
            model = GaussianProcessRegressor(random_state=configs["random_state"], alpha=configs['regression_config']['gp_alpha'],
                                           #n_restarts_optimizer=5, 
                                           kernel=kernel)

        else:
            print("Using default GPR parameters...")
            model = GaussianProcessRegressor(
                random_state=configs["random_state"],
                kernel=RationalQuadratic(length_scale=1.0, alpha=1.0)
            )
        
        pipe = Pipeline([
            ('scale_x', StandardScaler()),
            ('gpr', model)
        ])

        model = TransformedTargetRegressor(
            regressor=pipe,
            transformer=StandardScaler()
        )

    elif model_name is Regressor.extra_trees:
        model = ExtraTreesRegressor(
            random_state=configs["random_state"],
            n_estimators=100,  # You can tune this
            n_jobs=-1  # Use all cores
        )
    elif model_name is Regressor.ridge:
        # Ridge is sensitive to feature scales, so we pipeline a scaler
        model = make_pipeline(
            StandardScaler(),
            Ridge(random_state=configs["random_state"])
        )
    elif model_name is Regressor.random_forest_regressor:
        model = RandomForestRegressor(
            random_state=configs["random_state"],
            n_estimators=100,  # Using same default as ExtraTrees
            n_jobs=-1
        )
    elif model_name is Regressor.stacking:
        print("Defining stacked model...")
        # base models: regularized XGB, simple Ridge, and fast SVR
        estimators = [
            ('xgb', XGBRegressor(
                random_state=configs["random_state"],
                n_estimators=600,
                max_depth=5,
                min_child_weight=6,
                gamma=0.1,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=0.2,
                reg_lambda=2.0,
                learning_rate=0.02,
                eval_metric=configs['xgb_eval_metric'],
                verbosity=0
            )),
            ('extra_trees', ExtraTreesRegressor(
                n_estimators=200,
                max_depth=None,
                min_samples_split=4,
                random_state=configs["random_state"],
                n_jobs=-1
            )),
            ('random_forest', RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                min_samples_split=4,
                random_state=configs["random_state"],
                n_jobs=-1
            )),
            ('svr_linear', make_pipeline(
                StandardScaler(),
                SVR(kernel='linear', C=1.0)
            ))
        ]

        # meta-model combining predictions: simple robust Ridge model.
        final_estimator = XGBRegressor(
            random_state=configs["random_state"],
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_lambda=1.0,
            reg_alpha=0.2,
            verbosity=0
        )

        # cv=5 means it will use 5-fold cross-validation internally to generate predictions, which prevents data leakage.
        model = StackingRegressor(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5,  # Use 5 folds, 10 is too slow
            n_jobs=-1  # Use all cores
        )
    elif model_name is Regressor.neural_network:
        nn = NeuralNetRegressor(
            module=configs["nn_architecture"],
            **configs["nn_parameters"]
        )

        # Apply transformation to X
        pipe = Pipeline([
            ('scale_x', StandardScaler()),
            ('neural_net', nn),
        ])

        # Apply transformation for y
        model = TransformedTargetRegressor(
            regressor=pipe,
            transformer=StandardScaler()
        )

    if isinstance(model, XGBRegressor):
        x_train_sub, x_val_sub, y_train_sub, y_val_sub = train_test_split(
            X, y, test_size=0.1, random_state=configs["random_state"]
        )
        model.fit(x_train_sub, y_train_sub, eval_set=[(x_val_sub, y_val_sub)], verbose=False)
    elif model_name is Regressor.neural_network:
        # skorch / scikit-learn estimators expect NumPy arrays. Ensure float32 to avoid
        # dtype mismatches between inputs and model parameters (Double vs Float).
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        model.fit(X, y)
    else:
        model.fit(X, y)

    if 'regression_params' in configs:
        del configs['regression_params']  # clean up tuned params to not break next run

    return model


def train_model(X, y, i=None):
    """Run training pipeline. Returns processed data to calculate train score.
    ...
    """
    imputer = imputation(X, i)
    X_imp = imputer.transform(X)

    detector = outlier_detection(X_imp, y)
    train_mask = detector(X_imp)
    X_filt = X_imp[train_mask, :]
    y_proc = y[train_mask]
    print(f"Outlier detection: Kept {X_filt.shape[0]} / {X_imp.shape[0]} samples")

    # === THIS IS THE FIX ===
    # The logic is now simple: if it's a tree model, skip selection.
    model_name = configs["regression_method"]
    if not configs['selection_is_enabled'] and model_name in [Regressor.xgb, Regressor.extra_trees, Regressor.random_forest_regressor]:
        print("Using PassthroughSelector (skipping feature selection for tree-based model).")
        selection = PassthroughSelector()
        X_proc = selection.fit_transform(X_filt)
        #scaler = StandardScaler()
        #X_proc = scaler.fit_transform(X_proc)
        print(f"Selected features: {X_proc.shape[1]} (all)")

    # avoid feature selection for tree-based models
    elif model_name in [Regressor.xgb, Regressor.extra_trees, Regressor.random_forest_regressor]:
        print(f"Skipping feature selection for {model_name.name} (tree-based or stacking). Using PassthroughSelector.")
        selection = PassthroughSelector()
        X_proc = selection.fit_transform(X_filt)
        print(f"Selected features: {X_proc.shape[1]} (all)")

    else:
        # This will now correctly run for Ridge or Stacking
        print("Running feature_selection pipeline for non-tree model...")
        selection = feature_selection(X_filt, y_proc,
                                      thresh_var=configs['selection_thresh_var'],
                                      thresh_corr=configs['selection_thresh_corr'],
                                      rf_max_feats=configs['selection_rf_max_feats'],
                                      percentile=configs['selection_percentile']
                                      )
        X_proc = selection.transform(X_filt)
        print(f"Selected features: {X_proc.shape[1]}")
    # =======================

    model = fit(X_proc, y_proc)
    return imputer, detector, selection, model, X_proc, y_proc


def run_cv_experiment(x_data, y_data):
    """Runs a single K-fold CV with current global 'configs'.

    Parameters:
    ----------
    x_data: Features for training and validation
    y_data: Labels for training and validation

    Returns:
    ----------
    cv_df: A DataFrame with detailed results for each fold.
    """
    model_name = configs["regression_method"]
    outlier_method = configs['outlier_method']
    cv_results_list = []
    print(f"\n--- 🚀 Running CV for: {model_name.name} + {outlier_method.name} ---")
    folds = KFold(n_splits=configs["folds"], shuffle=True, random_state=configs["random_state"])

    for i, (train_index, validation_index) in enumerate(folds.split(x_data)):
        print(f"\n--- Fold {i} ---")
        x_val = x_data[validation_index, :]
        y_val = y_data[validation_index]
        x_train = x_data[train_index, :]
        y_train = y_data[train_index]

        # pipeline to fit on training set: x_proc, y_proc are processed training data
        imputer, detector, selection, model, x_proc, y_proc = train_model(x_train, y_train, i)
        if configs["regression_method"] is Regressor.neural_network:
            x_proc = np.asarray(x_proc, dtype=np.float32)
        y_train_pred = model.predict(x_proc)
        train_score = r2_score(y_proc, y_train_pred)

        # Validation Pipeline
        x_val_imputed = imputer.transform(x_val)
        val_mask = detector(x_val_imputed)
        x_val_filt = x_val_imputed[val_mask, :]
        y_val = y_val[val_mask]
        x_val_selected = selection.transform(x_val_filt)
        if configs["regression_method"] is Regressor.neural_network:
            x_val_selected = np.asarray(x_val_selected, dtype=np.float32)
        y_val_pred = model.predict(x_val_selected)
        val_score = r2_score(y_val, y_val_pred)
        print(f"Fold {i}: Train R² = {train_score:.4f}, Validation R² = {val_score:.4f}")

        cv_results_list.append({
            "model": model_name.name,
            "outlier_method": outlier_method.name,
            "fold": i,
            "train_score": train_score,
            "validation_score": val_score
        })

    return pd.DataFrame(cv_results_list)

