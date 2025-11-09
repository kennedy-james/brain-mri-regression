"""
Train models.
"""
import pandas as pd
import numpy as np
import torch
import torch.optim as opt
import torch.nn as nn
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, StackingRegressor, BaggingRegressor
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
from pytorch_tabnet.tab_model import TabNetRegressor
from skorch.dataset import Dataset
from catboost import CatBoostRegressor, Pool
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from skorch.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split


from cosinecows.config import configs, Regressor
from cosinecows.feature_selection import PassthroughSelector, feature_selection
from cosinecows.imputation import imputation
from cosinecows.outlier_detection import outlier_detection
    
class EarlyStopWrapper(BaseEstimator, RegressorMixin):
    """Wrapper that adds early stopping to compatible regressors (like XGB or CatBoost)."""
    def __init__(self, estimator, early_stopping_rounds=20, val_size=0.025, random_state=42):
        self.estimator = estimator.set_params(early_stopping_rounds=early_stopping_rounds)
        self.early_stopping_rounds = early_stopping_rounds
        self.val_size = val_size
        self.random_state = random_state
    def fit(self, X, y):
        X, y = check_X_y(X, y, multi_output=True)
        self.n_features_in_ = X.shape[1]
            
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.val_size, random_state=self.random_state
        )
    
        self.estimator.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        # Check if early stopping was triggered
        best_iter = self.estimator.best_iteration if isinstance(self.estimator, XGBRegressor) else self.estimator.best_iteration_ if isinstance(self.estimator, CatBoostRegressor) else None
        max_iter = self.estimator.n_estimators if isinstance(self.estimator, XGBRegressor) else self.estimator.get_params()['iterations'] if isinstance(self.estimator, CatBoostRegressor) else None
        
        if best_iter is not None and max_iter is not None:
            if best_iter >= max_iter - 1:
                print("❌❌❌ Early stopping not triggered!")
        self.is_fitted_ = True
        return self
    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X = check_array(X)
        return self.estimator.predict(X)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "regressor"
        tags.non_deterministic = True
        tags.target_tags.required = True
        tags.target_tags.multi_output = True
        return tags
    def get_params(self, deep=True):
        params = super().get_params(deep=False)
        if deep and hasattr(self.estimator, "get_params"):
            for k, v in self.estimator.get_params(deep=True).items():
                params[f"estimator__{k}"] = v
        return params
    def set_params(self, **params):
        est_params = {}
        for k, v in list(params.items()):
            if k.startswith("estimator__"):
                est_params[k[len("estimator__"):]] = v
                params.pop(k)
        for k, v in params.items():
            setattr(self, k, v)
        if est_params:
            self.estimator.set_params(**est_params)
        return self
    
class Float32Dataset(Dataset):
    def __init__(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        if y is not None:
            y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        super().__init__(X, y)

def build_nn(X):
    input_dim = X.shape[1]
    nn_depth      = 1
    nn_dropout    = [0.06964138137676289, 0.24215516087193295]
    nn_width      = [input_dim, 36, 1]
    nn_activation = ["nn.LeakyReLU", "nn.SELU"]
    nn_optimizer  = 'opt.Adamax'
    nn_loss       = 'nn.MSELoss'
    nn_parameters = {
        'batch_size': 35,
        'train_split': None,
        'lr': 0.01214788064320746,
        'max_epochs': 14
    }
    network_architecture = []
    for i in range(nn_depth + 1):
        network_architecture.append(nn.Dropout(nn_dropout[i]))
        network_architecture.append(nn.Linear(nn_width[i], nn_width[i + 1]))
        network_architecture.append(eval(nn_activation[i])())
        if i != nn_depth:                     # BatchNorm only between layers
            network_architecture.append(nn.BatchNorm1d(nn_width[i + 1]))



    neural_network = NeuralNetRegressor(
        module=nn.Sequential(*network_architecture),
        **nn_parameters,
        criterion=eval(nn_loss),
        optimizer=eval(nn_optimizer),
        iterator_train__drop_last=True,
        dataset=Float32Dataset
    )

    nn_pipe = Pipeline([
        ('scale_x', StandardScaler()),
        ('neural_net', neural_network),
    ])
    nn_model = TransformedTargetRegressor(
        regressor=nn_pipe,
        transformer=StandardScaler()
    )
    return nn_model



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

    if model_name is Regressor.stacking:
        print("Defining stacked model...")
        model = None
        estimators = None

        estimators = [
            ('svr', SVR(
                C=86, 
                epsilon=0.11
            )),
            ('xgb', XGBRegressor(
                    n_estimators=2410,
                    max_depth=5,
                    min_child_weight=13,
                    gamma=2.372524993310688,
                    subsample=0.7462741587810254,
                    colsample_bytree=0.6076272376281038,
                    reg_alpha=1.3240662642357892,
                    reg_lambda=2.586392652975843,
                    learning_rate=0.03332460602580017,
                    random_state=configs["random_state"]
            )),
            #('xgb', XGBRegressor(
            #        n_estimators=2410,
            #        max_depth=5,
            #        min_child_weight=13,
            #        gamma=2.372524993310688,
            #        subsample=0.7462741587810254,
            #        colsample_bytree=0.6076272376281038,
            #        reg_alpha=1.3240662642357892,
            #        reg_lambda=2.586392652975843,
            #        learning_rate=0.03332460602580017,
            #        random_state=configs["random_state"]
            #)),
            #('xgb', EarlyStopWrapper(
            #    XGBRegressor(
            #        n_estimators=10000,
            #        max_depth=9,
            #        min_child_weight=14,
            #        gamma=1.4692138346993904,
            #        subsample=0.5242435898389789,
            #        colsample_bytree=0.7983736226513591,
            #        reg_alpha=1.0788677992397164,
            #        reg_lambda=2.1877404829230365,
            #        learning_rate=0.0018296906700668437,
            #        random_state=configs["random_state"]
            #    ),
            #    early_stopping_rounds=300
            #)),
            ('gp', GaussianProcessRegressor(
                random_state=configs["random_state"], 
                alpha=2.965074241784881e-09, #configs['gp_alpha'],
                kernel=RationalQuadratic(
                    length_scale=6.124209435262154, #configs['gp_kernel_length_scale'],
                    alpha=0.669737299146556, #configs['gp_kernel_alpha']
                )
            )),
            ('bagging_svr', BaggingRegressor(
                estimator=SVR(C=88, epsilon=0.09),
                random_state=configs["random_state"],
            )),
            ('nn', build_nn(X)),

            #('catboost', EarlyStopWrapper(
            #    CatBoostRegressor(
            #        iterations=3626,
            #        learning_rate=0.008013493547220914,
            #        depth=7,
            #        l2_leaf_reg=0.2530799357736654,
            #        #early_stopping_rounds=158,
            #        random_strength=2.8241003601634453,
            #        bagging_temperature=1.4774574019375732,
            #        random_state=configs["random_state"],
            #        verbose=0
            #    ), 
            #    early_stopping_rounds=158
            #)),

        ]
        # cv=5 means it will use 5-fold cross-validation internally to generate predictions, which prevents data leakage.
        model = StackingRegressor(
            estimators=estimators,
            cv=5,  # Use 5 folds, 10 is too slow
            n_jobs=-1  # Use all cores
        )
    model.fit(X, y)

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
                                      percentile=configs['selection_percentile'],
                                      k_best=configs['selection_k_best'],
                                      )
        X_proc = selection.transform(X_filt)
        print(f"Selected features: {X_proc.shape[1]}")

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