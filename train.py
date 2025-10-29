import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, mutual_info_regression, VarianceThreshold, SelectFromModel
from sklearn.linear_model import Ridge # Used for imputation
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
import os.path
import joblib
from sklearn.experimental import enable_iterative_imputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler

#Lars imports
from xgboost import XGBRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from sklearn import pipeline
from sklearn.pipeline import make_pipeline
from sklearn import linear_model

# Set to 'True' to produce submission file for test data
FINAL_EVALUATION = False

# Reproducible dictionary defining experiment
configs = {
    "folds": 10,
    "random_state": 42,

    ## Possible impute methods (mean, median, most_frequent, KNN, iterative)
    "impute_method": "KNN",  # Imputation configuration
    'knn_neighbours': 75, # KNN configuration
    ## Possible neighbour weights for average (uniform, distance)
    'knn_weight': 'uniform', # KNN configuration
    "iterative_estimator": "Ridge()",  # Iterative configuration
    "iterative_iter": 1,  # Iterative configuration

    "regression_method": "ExtraTreesRegressor",

    "var_thresh": 0.01, "corr_thresh": 0.95, "xgb_thresh": 0.00001, "print_removed_ones": False,
    #variance #4 with 0, #59 with 0.008
    #correlation #12 with 0.999, #30 with 0.99, #37 with 0.98, 45 with 0.95, #53 with 0.9
}


def imputation(X, i):
    """Replace missing values in dataset using imputation

    Parameters
    ----------
    X: Dataset to learn imputation rule
    i: current CV iteration (for model loading)

    Returns
    ----------
    imputer: Trained imputer for imputing new data points
    """
    if configs["impute_method"] in ["mean", "median", "most_frequent"]:
        imputer = SimpleImputer(strategy=configs["impute_method"])
        imputer.fit(X)
    elif configs["impute_method"] == "KNN":
        scaler = StandardScaler()
        knn_imputer = KNNImputer()
        imputer = pipeline.make_pipeline(scaler, knn_imputer)
        imputer.fit(X)
    elif configs["impute_method"] == "iterative":
        # Avoid long training times by loading pretrained model (if possible)
        loadable_file = f'./models/imputers/{configs["iterative_estimator"].split('(')[0]}{configs["iterative_iter"]}_{i}.pkl'
        if i != None and os.path.isfile(loadable_file):
            imputer = joblib.load(loadable_file)
        else:
            imputer = IterativeImputer(
                random_state=configs["random_state"],
                estimator=eval(configs["iterative_estimator"]),
                max_iter=configs["iterative_iter"],
            )
            imputer.fit(X)

            joblib.dump(imputer, loadable_file)

    return imputer


def outlier_detection(X, y):
    """Detect outlier data points / samples that are to be removed

    Parameters
    ----------
    X: Features on which to train outlier prediction
    y: Labels for associated features

    Returns
    ----------
    detector: Detector that returns indices of outliers that should be deleted
    """

    # TODO: Replace detector with one that returns indices that are supposed to be deleted
    detector = lambda x: x
    return detector

class CorrelationRemover(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.to_drop_ = None

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        corr_matrix = df.corr(numeric_only=True).abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.to_drop_ = [col for col in upper_triangle.columns if any(upper_triangle[col] > self.threshold)]
        return self

    def transform(self, X):
        if self.to_drop_ is None:
            raise ValueError("Must fit before transform.")
        df = pd.DataFrame(X)
        return df.drop(columns=self.to_drop_).values


def feature_selection(x_train, y_train, varicance_threshold=0.01, correlation_threshold=0.95, mi_k=200, rf_max_features=120, rf_n_estimators=70):
    if hasattr(x_train, 'values'):
        x_train = x_train.values
        print("Converted to numpy array")

    rf = RandomForestRegressor(n_estimators=rf_n_estimators, random_state=42, n_jobs=-1)
    rf_selector = SelectFromModel(rf, max_features=rf_max_features, threshold='0.1*mean')

    selection = make_pipeline(
        # Low variance removal
        VarianceThreshold(threshold=configs["var_thresh"]), # 0.01
        # High correlation removal
        CorrelationRemover(threshold=configs["corr_thresh"]), # 0.95
        # Univariate feature selection
        SelectKBest(score_func=mutual_info_regression, k=min(200, x_train.shape[1])),
        # Non-linear embedded selection (RF instead of Lasso)
        rf_selector
    )
    selection.fit(x_train, y_train)
    return selection


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
    # TODO: Implement effective regression model
    #model = ExtraTreesRegressor(random_state=42)
    #model.fit(X, y)

    #model = XGBRegressor(random_state=configs["random_state"], n_estimators=100, verbosity=0)
    # Split 20% for early stopping (internal val)
    
    # Balanced reg from best run + capacity boost
    model = XGBRegressor(
        random_state=configs["random_state"],
        n_estimators=250,  # +50 for more stable fitting
        max_depth=4,
        min_child_weight=10,
        gamma=0.5,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.3,  # Mild L1 for feature sparsity
        reg_lambda=1.5,  # Moderate L2 for smoothness
        learning_rate=0.05,
        verbosity=0
    )
    model.fit(X, y)
    #model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    #model.fit(X, y)

    return model


def train_model(X, y, i=None):
    """Run training pipeline

    Parameters
    ----------
    X: Training data
    y: Labels to learn correct prediction
    i: current CV iteration (for model loading)

    Returns
    ----------
    imputer: Trained imputation model
    detector: Trained detector model
    selection: Trained selection model
    model: Trained prediction model
    X: Manipulated training data
    y: Manipulated training labels
    """
    imputer = imputation(X, i)
    X = imputer.transform(X)

    detector = outlier_detection(X, y)
    X = detector(X)

    selection = feature_selection(X, y)
    X = selection.transform(X)
    print(f"Selected features: {X.shape[1]}")

    model = fit(X, y)

    return imputer, detector, selection, model, X, y


if __name__ == "__main__":
    # Load the dataset for model training
    x_training_data = pd.read_csv("./data/X_train.csv", skiprows=1, header=None).values[
        :, 1:
    ]
    y_training_data = (
        pd.read_csv("./data/y_train.csv", skiprows=1, header=None).values[:, 1:].ravel()
    )

    if not FINAL_EVALUATION:
        # Use wandb to manage experiments
        with wandb.init(
            project="AML_task1",
            config=configs,
            tags=["regression"],
            name="regressor " + configs["regression_method"],
            notes="SelectKBest(mutual_info_regression, k=100).fit(X, y)",
        ) as run:
            # Apply KFold CV for model selection
            cv_stats = {"train_score": [], "validation_score": []}
            folds = KFold(n_splits=configs["folds"])
            for i, (train_index, validation_index) in enumerate(
                folds.split(x_training_data)
            ):
                x_val = x_training_data[validation_index, :]
                y_val = y_training_data[validation_index]
                x_train = x_training_data[train_index, :]
                y_train = y_training_data[train_index]

                # Pipeline to fit on training set
                imputer, detector, selection, model, x_train, y_train = train_model(
                    x_train, y_train, i
                )
                y_train_pred = model.predict(x_train)

                # Pipeline to perform predictions on validation set
                x_val = imputer.transform(x_val)
                x_val = detector(x_val)
                x_val = selection.transform(x_val)

                y_val_pred = model.predict(x_val)

                # Evaluate the model on training and validation sets
                train_score = r2_score(y_train, y_train_pred)
                val_score = r2_score(y_val, y_val_pred)
                print(f"Fold {i}: Train R² = {train_score:.4f}, Validation R² = {val_score:.4f}")

                cv_stats["train_score"].append(train_score)
                cv_stats["validation_score"].append(val_score)

            # Generate boxplots
            cv_df = pd.DataFrame(cv_stats)
            fig, ax = plt.subplots(figsize=(11, 13))
            sns.boxplot(data=cv_df, ax=ax)
            ax.set_title("Cross-Validation Results")
            ax.set_ylabel("R² Score")
            ax.set_xlabel("Score Type")
            run.log({"CV_Boxplot": wandb.Image(fig)})
            plt.close(fig)

            # Store raw CV results in table
            cv_table = wandb.Table(dataframe=cv_df)
            run.log({"CV Results": cv_table})

            # Log summary statistics
            run.summary["mean_train_score"] = np.mean(cv_stats["train_score"])
            run.summary["mean_validation_score"] = np.mean(cv_stats["validation_score"])
            run.summary["std_train_score"] = np.std(cv_stats["train_score"])
            run.summary["std_validation_score"] = np.std(cv_stats["validation_score"])
    else:
        x_test = pd.read_csv("./data/X_test.csv", skiprows=1, header=None).values[:, 1:]
        x_train = x_training_data
        y_train = y_training_data

        # Pipeline to fit on training set
        imputer, detector, selection, model, _, _ = train_model(x_train, y_train)

        # Pipeline to perform predictions on test set
        x_test = imputer.transform(x_test)
        x_test = detector(x_test)

        x_test = selection.transform(x_test)

        y_test_pred = model.predict(x_test)

        # Save predictions to submission file with the given format
        table = pd.DataFrame(
            {"id": np.arange(0, y_test_pred.shape[0]), "y": y_test_pred.flatten()}
        )
        table.to_csv("./submission.csv", index=False)
