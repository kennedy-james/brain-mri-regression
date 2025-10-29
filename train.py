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

# Lars imports
from xgboost import XGBRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from sklearn import pipeline
from sklearn.pipeline import make_pipeline
from sklearn import linear_model

# Jef imports
from sklearn.ensemble import IsolationForest
# Import for KNN outlier detection
from pyod.models.knn import KNN

# Set to 'True' to produce submission file for test data
FINAL_EVALUATION = False

# Reproducible dictionary defining experiment
OUTLIER_DETECTORS = ['zscore', 'knn', 'isolationForest']
# === NEW: Define models to test ===
REGRESSION_METHODS = ["ExtraTreesRegressor", "XGBRegressor"]

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

    "regression_method": REGRESSION_METHODS[0],
    "var_thresh": 0.01, "corr_thresh": 0.95, "xgb_thresh": 0.00001, "print_removed_ones": False,
    #variance #4 with 0, #59 with 0.008
    #correlation #12 with 0.999, #30 with 0.99, #37 with 0.98, 45 with 0.95, #53 with 0.9
    "outlier_detection": OUTLIER_DETECTORS[0],
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
    detector: Detector that removes from the data set the outlier samples
    """
    if configs['outlier_detection'] == OUTLIER_DETECTORS[0]:  # z-score
        from scipy import stats
        threshold = 3

        print(f"Using z-score detector (stateful, mean-based, threshold={threshold})")
        mean_train = np.nanmean(X, axis=0)
        std_train = np.nanstd(X, axis=0)
        std_train[std_train == 0] = 1.0

        def get_keep_mask(X_data):
            zscores = np.abs((X_data - mean_train) / std_train)
            mean_zscore_per_sample = np.mean(zscores, axis=1)
            keep_mask = (mean_zscore_per_sample <= threshold)

            if np.sum(keep_mask) == 0:
                print(f"WARNING: Z-score (mean) with threshold={threshold} removed all samples.")
                print("Fallback: Keeping all samples to prevent crash.")
                return np.ones(X_data.shape[0], dtype=bool)
            return keep_mask

    elif configs['outlier_detection'] == OUTLIER_DETECTORS[1]:  # KNN (STATEFUL)
        clf = KNN(contamination=0.05)
        clf.fit(X)
        print(f"Using KNN detector (stateful, contamination={clf.contamination})")

        def get_keep_mask(X_data):
            yhat = clf.predict(X_data)
            keep_mask = (yhat == 0)

            if np.sum(keep_mask) == 0:
                print("WARNING: KNN detector removed all samples. Keeping all as fallback.")
                return np.ones(X_data.shape[0], dtype=bool)
            return keep_mask

    elif configs['outlier_detection'] == OUTLIER_DETECTORS[2]:  # Isolation Forest (STATEFUL)
        iso = IsolationForest(contamination=0.05, random_state=configs["random_state"])
        iso.fit(X)
        print(f"Using IsolationForest (stateful, contamination={iso.contamination})")

        def get_keep_mask(X_data):
            yhat = iso.predict(X_data)
            keep_mask = (yhat == 1)

            if np.sum(keep_mask) == 0:
                print("WARNING: IsolationForest removed all samples. Keeping all as fallback.")
                return np.ones(X_data.shape[0], dtype=bool)
            return keep_mask

    return get_keep_mask


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

    model_name = configs["regression_method"]
    print(f"Fitting model: {model_name}")

    if model_name == "XGBRegressor":
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
    elif model_name == "ExtraTreesRegressor":
        # Using settings from your original code, but linked to config random_state
        model = ExtraTreesRegressor(
            random_state=configs["random_state"],
            n_estimators=100,  # You can tune this
            n_jobs=-1  # Use all cores
        )
    else:
        raise ValueError(f"Unknown regression_method: {model_name}")

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
    X_imputed = imputer.transform(X)
    get_keep_mask = outlier_detection(X_imputed, y)
    train_mask = get_keep_mask(X_imputed)
    X_filtered = X_imputed[train_mask, :]
    y_filtered = y[train_mask]
    selection = feature_selection(X, y)
    X_filtered_selected = selection.transform(X_filtered)
    print(f"Selected features: {X_filtered_selected.shape[1]}")
    model = fit(X_filtered_selected, y_filtered)
    return imputer, get_keep_mask, selection, model, X_filtered_selected, y_filtered


if __name__ == "__main__":
    # Load the dataset for model training
    x_training_data = pd.read_csv("./data/X_train.csv", skiprows=1, header=None).values[
        :, 1:
    ]
    y_training_data = (
        pd.read_csv("./data/y_train.csv", skiprows=1, header=None).values[:, 1:].ravel()
    )

    if not FINAL_EVALUATION:

        # === MODIFIED: Main testing loop ===

        # 1. Initialize a list to store ALL results
        all_results_list = []

        # 2. Outer loop for regression models
        for model_name in REGRESSION_METHODS:
            print(f"\n==========================================")
            print(f"   Testing Model: {model_name}")
            print(f"==========================================")
            configs["regression_method"] = model_name

            # 3. Inner loop for outlier methods (as before)
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
                imputer, detector, selection, model, x_train, y_train = train_model(
                    x_train, y_train, i
                )
                y_train_pred = model.predict(x_train)

                    y_train_pred = model.predict(x_train_final)
                    train_score = r2_score(y_train_final, y_train_pred)

                    x_val_imputed = imputer.transform(x_val)
                    x_val_selected = selection.transform(x_val_imputed)
                    y_val_pred = model.predict(x_val_selected)
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
        print("\n\n--- 📊 Final Performance Summary ---")

        # 5. Convert list of dicts to DataFrame
        results_df = pd.DataFrame(all_results_list)

        # 6. Save all results to CSV
        csv_filename = "all_model_results.csv"
        results_df.to_csv(csv_filename, index=False)
        print(f"\n✅ All results saved to '{csv_filename}'")

        # 7. Print summary statistics to console
        print("\n--- Validation R² Summary ---")
        # Group by model and outlier method, then show stats for validation_score
        summary_stats = results_df.groupby(['model', 'outlier_method'])['validation_score'].describe()
        print(summary_stats)

        # --- Generate boxplot (grouped by model) ---
        fig, ax = plt.subplots(figsize=(12, 8))

        # 8. Update plot to show grouped boxplots
        sns.boxplot(data=results_df, x='outlier_method', y='validation_score', hue='model', ax=ax)

        ax.set_title("Model Comparison by Outlier Method (Validation R²)")
        ax.set_ylabel("R² Score (Validation)")
        ax.set_xlabel("Outlier Detection Method")
        ax.legend(title="Model")

        plot_filename = "model_comparison_boxplot.png"
        plt.savefig(plot_filename)
        print(f"\n✅ Saved grouped boxplot to '{plot_filename}'")
        # plt.show()

    else:
        # --- FINAL_EVALUATION = True ---
        # This block now correctly uses the 'fit' function,
        # which will respect the "regression_method" set in the 'configs' dict.
        # Before running this, you should manually set your *best* combination
        # in the 'configs' dict at the top of the file.

        print(f"🚀 Running final evaluation pipeline with:")
        print(f"   Model: {configs['regression_method']}")
        print(f"   Outlier Detector: {configs['outlier_detection']}")

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
        print("\n✅ Successfully generated submission.csv")