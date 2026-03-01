"""
Train models.
"""
import pandas as pd
import numpy as np
import torch
import torch.optim as opt
import torch.nn as nn
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, StackingRegressor, BaggingRegressor, VotingRegressor
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
#from catboost import CatBoostRegressor, Pool


from cosinecows.config import configs, Regressor
from cosinecows.feature_selection import PassthroughSelector, feature_selection, feature_selection_old
from cosinecows.imputation import imputation
from cosinecows.outlier_detection import outlier_detection

class Float32Dataset(Dataset):
    def __init__(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        if y is not None:
            y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        super().__init__(X, y)

def build_nn(X):
    input_dim = 53
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

    #fs_pipe = feature_selection(
    #    score_func=configs['regression_params']['score_func'],
    #    k_best=configs['regression_params']['k_best'],
    #)

    fs_pipe_old = feature_selection_old(
        k_best=194,
    )

    #if model_name is Regressor.xgb:
    #    # check if optuna provided set of tuned params
    #    if 'regression_params' in configs:
    #        print("Using tuned XGBoost parameters from Optuna...")
    #        model = XGBRegressor(**configs['regression_params'])
    #    else:
    #        model = XGBRegressor(
    #            random_state=configs["random_state"],
    #            n_estimators=300,
    #            max_depth=5,  # shallower trees
    #            min_child_weight=15,  # require more samples per leaf
    #            gamma=1.0,  # stronger split penalty
    #            subsample=0.75,
    #            colsample_bytree=0.4,  # fewer features per tree
    #            reg_alpha=0.5,
    #            reg_lambda=3.0,
    #            learning_rate=0.03,
    #            eval_metric=configs['xgb_eval_metric'],
    #            early_stopping_rounds=configs['xgb_early_stopping_rounds'],
    #            verbosity=0
    #        )
    #elif model_name is Regressor.gaussian_process:
#
    #    if 'regression_params' in configs:
    #        print("Using tuned GPR parameters from Optuna...")
    #        kernel = RationalQuadratic(length_scale=configs['regression_config']['length_scale'],
    #                                   alpha=configs['regression_config']['alpha'])
    #        model = GaussianProcessRegressor(random_state=configs["random_state"], alpha=configs['regression_config']['gp_alpha'],
    #                                       #n_restarts_optimizer=5, 
    #                                       kernel=kernel)
#
    #    else:
    #        print("Using default GPR parameters...")
    #        model = GaussianProcessRegressor(
    #            random_state=configs["random_state"],
    #            kernel=RationalQuadratic(length_scale=1.0, alpha=1.0)
    #        )
    #    
    #    pipe = Pipeline([
    #        ('scale_x', StandardScaler()),
    #        ('gpr', model)
    #    ])
#
    #    model = TransformedTargetRegressor(
    #        regressor=pipe,
    #        transformer=StandardScaler()
    #    )
#
    #elif model_name is Regressor.extra_trees:
    #    model = ExtraTreesRegressor(
    #        random_state=configs["random_state"],
    #        **configs['xtrees_parameters'],
    #        n_jobs=-1  # Use all cores
    #    )
    #elif model_name is Regressor.svr:
    #    model = make_pipeline(
    #        StandardScaler(),
    #        SVR(
    #            kernel=configs['svr_kernel'],
    #            C=configs['svr_C'],
    #            epsilon=configs['svr_epsilon'],
    #            gamma=configs['svr_gamma']
    #        )
    #    )
    #elif model_name is Regressor.ridge:
    #    # Ridge is sensitive to feature scales, so we pipeline a scaler
    #    model = make_pipeline(
    #        StandardScaler(),
    #        Ridge(random_state=configs["random_state"])
    #    )
    #elif model_name is Regressor.random_forest_regressor:
    #    model = RandomForestRegressor(
    #        random_state=configs["random_state"],
    #        n_estimators=100,  # Using same default as ExtraTrees
    #        n_jobs=-1
    #    )

    if model_name is Regressor.stacking:
        print("Defining stacked model...")
        ##############################
        xgb_f = feature_selection(
            score_func='f_regression',
            k_best=201,
        )

        xgb_model= XGBRegressor(
                    n_estimators=4563,
                    max_depth=6,
                    min_child_weight=13,
                    gamma=1.087304331334667,
                    subsample=0.4598613844539232,
                    colsample_bytree=0.8360514902773765,
                    reg_alpha=0.5009518183656549,
                    reg_lambda=2.9324605767035323,
                    learning_rate=0.0066215147930692225,
                    random_state=configs["random_state"]
            )
        xgb_pipeline = Pipeline([
            ('feature_selection', xgb_f),
            ('xgb', xgb_model)
        ])
        #################################
        gpr_f = feature_selection(
            score_func='f_regression',
            k_best=205,
        )

        gpr_model = GaussianProcessRegressor(
                    random_state=configs["random_state"], 
                    alpha=3.935643578586644e-10, #configs['gp_alpha'],
                    kernel=RationalQuadratic(
                        length_scale=6.240171100411289, #configs['gp_kernel_length_scale'],
                        alpha=0.7052022185369317, #configs['gp_kernel_alpha']
                    )
            )
        
        gp_pipeline = Pipeline([
            ('feature_selection', fs_pipe_old),
            ('standard_scaler', StandardScaler()),
            ('gpr', gpr_model)
        ])

        svr_model = SVR(
                C=86, 
                epsilon=0.11
            )
        svr_pipeline = Pipeline([
            ('feature_selection', fs_pipe_old),
            ('standard_scaler', StandardScaler()),
            ('svr', svr_model)
        ])


        svr_bagging_f = fs_pipe_old
        svr_bagging_model = BaggingRegressor(
                estimator=SVR(C=88, epsilon=0.09),
                random_state=configs["random_state"],
            )
        svr_bagging_pipeline = Pipeline([
            ('feature_selection', fs_pipe_old),
            ('standard_scaler', StandardScaler()),
            ('bagging_svr', svr_bagging_model)
        ])
        ###############################
        nn_f = feature_selection(
            score_func='random_forest_regressor',
            k_best=53,
        )

        nn_model = build_nn(X)

        nn_pipeline = Pipeline([
            ('feature_selection', nn_f),
            ('nn', nn_model)
        ])


        estimators = [
            ('xgb', xgb_pipeline),
    
            ('gp', gp_pipeline),

            ('svr', svr_pipeline),

            #('xgb', XGBRegressor(
            #    n_estimators=10000,
            #    max_depth=9,
            #    min_child_weight=14,
            #    gamma=1.4692138346993904,
            #    subsample=0.5242435898389789,
            #    colsample_bytree=0.7983736226513591,
            #    reg_alpha=1.0788677992397164,
            #    reg_lambda=2.1877404829230365,
            #    learning_rate=0.0018296906700668437,
            #    random_state=configs["random_state"]
            #)),


            ('bagging_svr', svr_bagging_pipeline),
            ('nn', nn_pipeline),

            #('catboost', CatBoostRegressor(
            #    iterations=3626,
            #    learning_rate=0.008013493547220914,
            #    depth=7,
            #    l2_leaf_reg=0.2530799357736654,
            #    early_stopping_rounds=158,
            #    random_strength=2.8241003601634453,
            #    bagging_temperature=1.4774574019375732,
            #    random_state=configs["random_state"],
            #    verbose=0
            #)),
            #regression_config = {
            #'catboost_parameters': {
            #    'iterations': 3626, # Should be much greater than 100
            #    'learning_rate': 0.008013493547220914, # Should be less than 0.7
            #    'depth': 7, # sometimes good to use 10
            #    'l2_leaf_reg': 0.2530799357736654,
            #    'early_stopping_rounds': 158, # at least 50 for several thousand iterations
            #    'random_strength': 2.8241003601634453,
            #    'bagging_temperature': 1.4774574019375732
            #}

        ]
        # base models: regularized XGB, simple Ridge, and fast SVR
        # estimators = [
        #     ('xgb', XGBRegressor(
        #         random_state=configs["random_state"],
        #         n_estimators=600,
        #         max_depth=5,
        #         min_child_weight=6,
        #         gamma=0.1,
        #         subsample=0.8,
        #         colsample_bytree=0.7,
        #         reg_alpha=0.2,
        #         reg_lambda=2.0,
        #         learning_rate=0.02,
        #         eval_metric=configs['xgb_eval_metric'],
        #         verbosity=0
        #     )),
        #     ('extra_trees', ExtraTreesRegressor(
        #         n_estimators=200,
        #         max_depth=None,
        #         min_samples_split=4,
        #         random_state=configs["random_state"],
        #         n_jobs=-1
        #     )),
        #     ('random_forest', RandomForestRegressor(
        #         n_estimators=200,
        #         max_depth=None,
        #         min_samples_split=4,
        #         random_state=configs["random_state"],
        #         n_jobs=-1
        #     )),
        #     ('svr_linear', make_pipeline(
        #         StandardScaler(),
        #         SVR(kernel='linear', C=1.0)
        #     ))
        # ]

        # meta-model combining predictions: simple robust Ridge model.
        # final_estimator = XGBRegressor(
        #     random_state=configs["random_state"],
        #     n_estimators=400,
        #     max_depth=4,
        #     learning_rate=0.05,
        #     subsample=0.8,
        #     colsample_bytree=0.7,
        #     reg_lambda=1.0,
        #     reg_alpha=0.2,
        #     verbosity=0
        # )


        # cv=5 means it will use 5-fold cross-validation internally to generate predictions, which prevents data leakage.
        model = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(),
            cv=5,  # Use 5 folds, 10 is too slow
            n_jobs=-1  # Use all cores
        )
    elif model_name is Regressor.neural_network:
        # Construct NN
        network_architecture = []

        for i in range(configs['nn_depth'] + 1):
            network_architecture.append(nn.Dropout(configs['nn_dropout'][i]))
            network_architecture.append(nn.Linear(configs['nn_width'][i], configs['nn_width'][i + 1]))
            network_architecture.append(eval(configs['nn_activation'][i])())
            if i != configs['nn_depth']:
                network_architecture.append(nn.BatchNorm1d(configs['nn_width'][i + 1]))
        
        # Define NN
        neural_network = NeuralNetRegressor(
            module=nn.Sequential(*network_architecture),
            **configs["nn_parameters"],
            criterion=eval(configs['nn_loss']),
            optimizer=eval(configs['nn_optimizer']),
            iterator_train__drop_last=True # Avoid crash when batch size 1
        )

        # Apply transformation to X
        pipe = Pipeline([
            ('scale_x', StandardScaler()),
            ('neural_net', neural_network),
        ])

        # Apply transformation for y
        model = TransformedTargetRegressor(
            regressor=pipe,
            transformer=StandardScaler()
        )
    elif model_name is Regressor.tab_net:
        model = TabNetRegressor(
            seed=configs['random_state'],
            optimizer_fn=eval(configs['optimizer_fn']),
            **configs['tab_parameters']
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
    elif model_name is Regressor.tab_net:
        y = y.reshape(-1, 1)
        model.fit(X, y, **configs['tab_fitting'])
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
    #if not configs['selection_is_enabled'] and model_name in [Regressor.xgb, Regressor.extra_trees, Regressor.random_forest_regressor]:
    #    print("Using PassthroughSelector (skipping feature selection for tree-based model).")
    #    selection = PassthroughSelector()
    #    X_proc = selection.fit_transform(X_filt)
    #    #scaler = StandardScaler()
    #    #X_proc = scaler.fit_transform(X_proc)
    #    print(f"Selected features: {X_proc.shape[1]} (all)")
#
    ## avoid feature selection for tree-based models
    #elif model_name in [Regressor.xgb, Regressor.extra_trees, Regressor.random_forest_regressor]:
    #    print(f"Skipping feature selection for {model_name.name} (tree-based or stacking). Using PassthroughSelector.")
    #    selection = PassthroughSelector()
    #    X_proc = selection.fit_transform(X_filt)
    #    print(f"Selected features: {X_proc.shape[1]} (all)")

    
    # This will now correctly run for Ridge or Stacking
    #print("Running feature_selection pipeline for non-tree model...")
    #selection = feature_selection(X_filt, y_proc,
    #                              thresh_var=configs['selection_thresh_var'],
    #                              thresh_corr=configs['selection_thresh_corr'],
    #                              rf_max_feats=configs['selection_rf_max_feats'],
    #                              percentile=configs['selection_percentile'],
    #                              k_best=configs['selection_k_best'],
    #                              )
    #
    #X_proc = selection.transform(X_filt)
    X_proc = X_filt
    selection = None
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
        x_val_selected = x_val_filt
        #x_val_selected = selection.transform(x_val_filt)
        #if configs["regression_method"] is Regressor.neural_network:
        #    x_val_selected = np.asarray(x_val_selected, dtype=np.float32)
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

