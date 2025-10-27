import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.linear_model import Ridge # Used for imputation
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
import os.path
import joblib
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

#Lars imports
from xgboost import XGBRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Jef imports
from sklearn.ensemble import IsolationForest

# Set to 'True' to produce submission file for test data
FINAL_EVALUATION = False

# Reproducible dictionary defining experiment
configs = {
    "folds": 10,
    "random_state": 42,

    ## Possible impute methods (mean, median, most_frequent, KNN, iterative)
    "impute_method": "mean",
    # 'knn_neighbours': 75, # KNN configuration
    ## Possible neighbour weights for average (uniform, distance)
    # 'knn_weight': 'uniform', # KNN configuration
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
        imputer = KNNImputer(
            n_neighbors=configs["knn_neighbours"], weights=configs["knn_weight"]
        )
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
    iso = IsolationForest(contamination=0.05, random_state=configs["random_state"])
    yhat = iso.fit_predict(X)
    detector = lambda X: X[yhat == 1, :]
    return detector

def xgb_feature_importance_selector(X, y, importance_thresh=0.0001):
    """
    Fits an XGBoost model and selects features based on importance threshold.
    Returns a boolean mask for keeping features above the threshold.
    """
    model = XGBRegressor(random_state=configs["random_state"], n_estimators=100, verbosity=0)
    model.fit(X, y)
    
    importances = model.feature_importances_
    keep_mask = importances > importance_thresh

    return keep_mask, importances

class FeatureSelector:
    """
    Feature selection to remove irrelevant or redundant features.
    
    Parameters
    ----------
    X: NumPy array of features on which to train feature selector
    y: NumPy array of labels for associated features
    var_thresh: Variance threshold for feature selection
    corr_thresh: Correlation threshold for feature selection
    xgb_thresh: XGBoost feature importance threshold
    
    Returns
    ----------
    selector: Trained feature selector that can be applied to other data points
    """
    def __init__(self, var_thresh=None, corr_thresh=None, xgb_thresh=None, print_removed_ones=False):
        self.var_thresh = var_thresh
        self.corr_thresh = corr_thresh
        self.xgb_thresh = xgb_thresh
        self.mask_ = None  # Will be set on fit
        self.print_removed_ones = print_removed_ones

    def fit(self, X, y):
        """
        Fit the selector: compute the mask based on thresholds.
        X and y are NumPy arrays.
        y is required if xgb_thresh is not None.
        Handles NaNs by ignoring them in variance and correlation calculations.
        """
        # Convert to DataFrame for easy column tracking
        df_original = pd.DataFrame(X)
        df_filtered = df_original.copy()

        # 1. Low-variance filter
        if self.var_thresh is not None:
            # Temporarily impute NaNs for variance calc
            df_for_var = df_filtered.copy()
            col_means = df_for_var.mean(numeric_only=True, skipna=True)
            df_for_var = df_for_var.fillna(col_means)
            
            selector = VarianceThreshold(threshold=self.var_thresh)
            var_mask = selector.fit(df_for_var).get_support()

            # Printing only: Variance filter removals
            n_removed_var = np.sum(~var_mask)
            print(f"Variance filter (thresh={self.var_thresh}): Removed {n_removed_var} features")
            if n_removed_var > 0 and self.print_removed_ones:
                dropped_mask_var = ~var_mask
                dropped_cols_var = df_original.columns[dropped_mask_var]
                dropped_vars = selector.variances_[dropped_mask_var]
                for col, var_val in zip(dropped_cols_var, dropped_vars):
                    print(f"  - Column {col}: variance = {var_val:.6f}")

            df_filtered = df_filtered.iloc[:, var_mask]

        # 2. High-correlation filter
        if self.corr_thresh is not None:
            corr_matrix = df_filtered.corr(numeric_only=True).abs()
            upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            highly_correlated_features_to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > self.corr_thresh)]
            corr_mask = ~df_filtered.columns.isin(highly_correlated_features_to_drop)

            # Printing only: Correlation filter removals
            n_removed_corr = len(highly_correlated_features_to_drop)
            print(f"Correlation filter (thresh={self.corr_thresh}): Removed {n_removed_corr} features")
            if n_removed_corr > 0 and self.print_removed_ones:
                for feat in highly_correlated_features_to_drop:
                    # Find the maximum correlation for this feature (with others)
                    max_corr_col = upper_triangle[feat].idxmax()
                    max_corr_val = upper_triangle[feat].max()
                    print(f"  - Column {feat}: max correlation = {max_corr_val:.4f} (with Column {max_corr_col})")

            df_filtered = df_filtered.loc[:, corr_mask]

        # 3. XGBoost importance filter
        if self.xgb_thresh is not None:
            # Convert back to NumPy for XGBoost
            X_for_xgb = df_filtered.values
            xgb_mask, importances = xgb_feature_importance_selector(X_for_xgb, y, importance_thresh=self.xgb_thresh)

            # Printing only: XGBoost filter removals (uses returned importances)
            dropped_mask_xgb = ~xgb_mask
            n_removed_xgb = np.sum(dropped_mask_xgb)
            print(f"XGBoost filter (thresh={self.xgb_thresh}): Removed {n_removed_xgb} features")
            if n_removed_xgb > 0 and self.print_removed_ones:
                dropped_cols_xgb = df_filtered.columns[dropped_mask_xgb]
                dropped_importances = importances[dropped_mask_xgb]
                for col, imp_val in zip(dropped_cols_xgb, dropped_importances):
                    print(f"  - Column {col}: importance = {imp_val:.6f}")

            df_filtered = df_filtered.iloc[:, xgb_mask]

        # Final mask: boolean over original indices (stacked via column tracking)
        self.mask_ = df_original.columns.isin(df_filtered.columns)

        # Final summary print
        print(f"✅ Selected {self.mask_.sum()} / {X.shape[1]} features "
              f"({self.mask_.sum()/X.shape[1]*100:.1f}%)")
        
        return self
    
    def transform(self, X):
        """
        Apply the fitted mask to select columns from X.
        Assumes X is a NumPy array with the same number of features as during fit.
        """
        if self.mask_ is None:
            raise ValueError("Must call fit before transform.")
        return X[:, self.mask_]


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

    selection = FeatureSelector(var_thresh=configs["var_thresh"], corr_thresh=configs["corr_thresh"], xgb_thresh=configs["xgb_thresh"], print_removed_ones=configs["print_removed_ones"])
    selection.fit(X, y)
    X = selection.transform(X)

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
