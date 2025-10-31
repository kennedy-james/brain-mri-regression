import os.path
import joblib
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.feature_selection import mutual_info_regression, SelectFromModel, SelectPercentile
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import Ridge  # Used for imputation AND regression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# Lars imports
from xgboost import XGBRegressor
from sklearn import pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline

# Jef imports
from enum import Enum, auto
from pyod.models.knn import KNN
from sklearn.ensemble import IsolationForest


class RunMode(Enum):
    FINAL_EVALUATION = auto() # produce submission file for test data
    WANDB = auto() # log to wandb
    CVRUN = auto()

RUN_MODE = RunMode.CVRUN

# Reproducible dictionary defining experiment
IMPUTERS = ['mean', 'median', 'most_frequent', 'KNN', 'iterative']
OUTLIER_DETECTORS = ['zscore', 'knn', 'isolationForest']
REGRESSORS = ['XGBRegressor','ExtraTreesRegressor', 'Ridge', 'RandomForestRegressor']

configs = {
    'folds': 10,
    'random_state': 42,
    'impute_method': IMPUTERS[4],
    'knn_neighbours': 75,
    'knn_weight': 'uniform',  # possible neighbour weights for average (uniform, distance)
    'iterative_estimator': 'Ridge()',  # Iterative configuration
    'iterative_iter': 1,  # Iterative configuration
    'outlier_detection': OUTLIER_DETECTORS[0],
    'regression_method': REGRESSORS[0],
    'selection': {'thresh_var': 0.01, 'thresh_corr': 0.95},
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
    method = configs["impute_method"]
    if method in IMPUTERS[:3]: # mean, median, most_frequent
        imputer = SimpleImputer(strategy=configs["impute_method"])
        imputer.fit(X)
    elif method == IMPUTERS[3]: # KNN
        scaler = StandardScaler()
        knn_imputer = KNNImputer(n_neighbors=configs["knn_neighbours"], weights=configs["knn_weight"])
        imputer = pipeline.make_pipeline(scaler, knn_imputer)
        imputer.fit(X)
    elif method == IMPUTERS[4]: # iterative imputer
        loadable_file = f'./models/imputers/{configs["iterative_estimator"].split("(")[0]}{configs["iterative_iter"]}_{i}.pkl'
        if i is not None and os.path.isfile(loadable_file):
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
    detector: Detector that returns indices of inliers that should be kept
    """
    def safe_detector(predict_fn):
        def wrapped(X_data):
            detector = predict_fn(X_data)
            if np.sum(detector) == 0:
                print(f"WARNING: {method} removed all samples. Keeping all as fallback.")
                return np.ones(X_data.shape[0], dtype=bool)
            return detector

        return wrapped

    method = configs['outlier_detection']
    if method == OUTLIER_DETECTORS[0]:  # z-score
        threshold = 3 # std devs
        print(f"Using z-score detector (stateful, mean-based, threshold={threshold})")
        mean_train = np.nanmean(X, axis=0)
        std_train = np.nanstd(X, axis=0)
        std_train[std_train == 0] = 1.0

        def predict_fn(X_data):
            zscores = np.abs((X_data - mean_train) / std_train)
            return np.mean(zscores, axis=1) <= threshold

        get_detector = safe_detector(predict_fn)

    elif method == OUTLIER_DETECTORS[1]:  # KNN
        clf = KNN(contamination=0.05)
        clf.fit(X)
        print(f"Using KNN detector (stateful, contamination={clf.contamination})")
        get_detector = safe_detector(lambda X_data: clf.predict(X_data) == 0) # inliers are labeled 0, outliers 1

    elif method == OUTLIER_DETECTORS[2]:  # Isolation Forest
        iso = IsolationForest(contamination=0.05, random_state=configs["random_state"])
        iso.fit(X)
        print(f"Using IsolationForest (stateful, contamination={iso.contamination})")
        get_detector = safe_detector(lambda X_data: iso.predict(X_data) == 1) # inliers are labeled 1, outliers -1

    return get_detector


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


def feature_selection(x_train, y_train, thresh_var=0.01, thresh_corr=0.95, rf_max_feats=120, rf_n_estimators=70):
    if hasattr(x_train, 'values'):
        x_train = x_train.values
        print("Converted to numpy array")

    rf = RandomForestRegressor(n_estimators=rf_n_estimators, random_state=configs['random_state'], n_jobs=-1)
    rf_selector = SelectFromModel(rf, max_features=rf_max_feats, threshold='0.1*mean')

    selection = make_pipeline(
        VarianceThreshold(threshold=thresh_var),  # low variance removal
        CorrelationRemover(threshold=thresh_corr),  # high correlation removal
        SelectPercentile(score_func=mutual_info_regression, percentile=24), # equivalent to KBest=200, more robust
        rf_selector # non-linear embedded selection (RF instead of Lasso)
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
    model_name = configs["regression_method"]
    print(f"Fitting model: {model_name}")

    if model_name == REGRESSORS[0]:  # XGBRegressor
        model = XGBRegressor(
            random_state=configs["random_state"],
            n_estimators=250,
            max_depth=4,
            min_child_weight=10,
            gamma=0.5,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.3,
            reg_lambda=1.5,
            learning_rate=0.05,
            verbosity=0
        )
    elif model_name == REGRESSORS[1]:  # ExtraTreesRegressor
        model = ExtraTreesRegressor(
            random_state=configs["random_state"],
            n_estimators=100,  # You can tune this
            n_jobs=-1  # Use all cores
        )
    elif model_name == REGRESSORS[2]: # Ridge
        # Ridge is sensitive to feature scales, so we pipeline a scaler
        model = make_pipeline(
            StandardScaler(),
            Ridge(random_state=configs["random_state"])
        )
    elif model_name == REGRESSORS[3]: # RandomForestRegressor
        model = RandomForestRegressor(
            random_state=configs["random_state"],
            n_estimators=100,  # Using same default as ExtraTrees
            n_jobs=-1
        )

    model.fit(X, y)
    return model


def train_model(X, y, i=None):
    """Run training pipeline. Returns processed data to calculate train score.

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
    X_imp = imputer.transform(X)

    detector = outlier_detection(X_imp, y)
    train_mask = detector(X_imp)
    X_filt = X_imp[train_mask, :]
    y_proc = y[train_mask]
    print(f"Outlier detection: Kept {X_filt.shape[0]} / {X_imp.shape[0]} samples")

    selection = feature_selection(X_filt, y_proc,
        thresh_var=configs['selection']['thresh_var'],
        thresh_corr=configs['selection']['thresh_corr']
    )
    X_proc = selection.transform(X_filt)
    print(f"Selected features: {X_proc.shape[1]}")

    model = fit(X_proc, y_proc)
    return imputer, detector, selection, model, X_proc, y_proc


if __name__ == "__main__":
    # Load the dataset for model training
    x_training_data = pd.read_csv("./data/X_train.csv", skiprows=1, header=None).values[
        :, 1:
    ]
    y_training_data = (
        pd.read_csv("./data/y_train.csv", skiprows=1, header=None).values[:, 1:].ravel()
    )

    if RUN_MODE == RunMode.CVRUN:

        # 1. Initialize a list to store ALL results
        all_results_list = []

        # 2. Outer loop for regression models (now has 4 models)
        for model_name in REGRESSORS:
            print(f"\n==========================================")
            print(f"   Testing Model: {model_name}")
            print(f"==========================================")
            configs["regression_method"] = model_name

            # 3. Inner loop for outlier methods
            for outlier_method in OUTLIER_DETECTORS:
                print(f"\n--- 🚀 Testing Outlier Method: {outlier_method} ---")
                configs["outlier_detection"] = outlier_method

                folds = KFold(n_splits=configs["folds"], shuffle=True, random_state=configs["random_state"])
                for i, (train_index, validation_index) in enumerate(
                        folds.split(x_training_data)
                ):
                    print(f"\n--- Fold {i} ---")
                    x_val = x_training_data[validation_index, :]
                    y_val = y_training_data[validation_index]
                    x_train = x_training_data[train_index, :]
                    y_train = y_training_data[train_index]

                    # Pipeline to fit on training set
                    # Note: x_train_final and y_train_final are the filtered/selected data
                    imputer, detector, selection, model, x_proc, y_proc = train_model(
                        x_train, y_train, i
                    )

                    # Get train score
                    y_train_pred = model.predict(x_proc)
                    train_score = r2_score(y_proc, y_train_pred)

                    # --- Validation Pipeline ---
                    # 1. Impute val data
                    x_val_imputed = imputer.transform(x_val)

                    # 2. Apply selection (NO outlier removal on validation data)
                    x_val_selected = selection.transform(x_val_imputed)

                    # 3. Predict
                    y_val_pred = model.predict(x_val_selected)

                    # 4. Score against original y_val
                    val_score = r2_score(y_val, y_val_pred)

                    print(f"Fold {i}: Train R² = {train_score:.4f}, Validation R² = {val_score:.4f}")

                    # 4. Append detailed results to the master list
                    all_results_list.append({
                        "model": model_name,
                        "outlier_method": outlier_method,
                        "fold": i,
                        "train_score": train_score,
                        "validation_score": val_score
                    })

        # --- Final Comparison and CSV Export ---
        print("\n\n--- Final Performance Summary ---")
        results_df = pd.DataFrame(all_results_list)

        csv_filename = "all_model_results.csv"
        results_df.to_csv(csv_filename, index=False)
        print(f"\n All results saved to '{csv_filename}'")

        print("\n--- Validation R² Summary ---")
        summary_stats = results_df.groupby(['model', 'outlier_method'])['validation_score'].describe()
        print(summary_stats)

        # --- Generate boxplot (grouped by model) ---
        fig, ax = plt.subplots(figsize=(14, 8))  # Made wider for 4 models
        sns.boxplot(data=results_df, x='outlier_method', y='validation_score', hue='model', ax=ax)

        ax.set_title("Model Comparison by Outlier Method (Validation R²)")
        ax.set_ylabel("R² Score (Validation)")
        ax.set_xlabel("Outlier Detection Method")
        ax.legend(title="Model")

        plot_filename = "model_comparison_boxplot.png"
        plt.savefig(plot_filename)
        print(f"\n Saved grouped boxplot to '{plot_filename}'")
        # plt.show()

    else:
        print(f" Running final evaluation pipeline with:")
        print(f"   Model: {configs['regression_method']}")
        print(f"   Outlier Detector: {configs['outlier_detection']}")

        x_test = pd.read_csv("./data/X_test.csv", skiprows=1, header=None).values[:, 1:]
        x_train = x_training_data
        y_train = y_training_data

        # Pipeline to fit on training set
        imputer, detector, selection, model, _, _ = train_model(x_train, y_train)

        # 1. Impute test data
        x_test_imputed = imputer.transform(x_test)
        # 2. Apply feature selection (NO outlier removal)
        x_test_selected = selection.transform(x_test_imputed)
        # 3. Predict
        y_test_pred = model.predict(x_test_selected)

        # Save predictions to submission file
        # The number of predictions should match the original x_test rows
        table = pd.DataFrame(
            {"id": np.arange(0, y_test_pred.shape[0]), "y": y_test_pred.flatten()}
        )
        table.to_csv("./submission.csv", index=False)
        print("\n✅ Successfully generated submission.csv")