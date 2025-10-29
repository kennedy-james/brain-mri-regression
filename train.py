import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.linear_model import Ridge  # Used for imputation
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
import os.path
import joblib
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

# Lars imports
from xgboost import XGBRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

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
    "impute_method": "mean",
    "iterative_estimator": "Ridge()",  # Iterative configuration
    "iterative_iter": 1,  # Iterative configuration

    # This will be set by the loop in __main__
    "regression_method": REGRESSION_METHODS[0],
    "var_thresh": 0.01, "corr_thresh": 0.95, "xgb_thresh": 0.00001, "print_removed_ones": False,

    # This will be set by the loop in __main__
    "outlier_detection": OUTLIER_DETECTORS[0],
}


def imputation(X, i):
    # (No changes to this function)
    """Replace missing values in dataset using imputation..."""
    if configs["impute_method"] in ["mean", "median", "most_frequent"]:
        imputer = SimpleImputer(strategy=configs["impute_method"])
        imputer.fit(X)
    elif configs["impute_method"] == "KNN":
        imputer = KNNImputer(
            n_neighbors=configs["knn_neighbours"], weights=configs["knn_weight"]
        )
        imputer.fit(X)
    elif configs["impute_method"] == "iterative":
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
    # (No changes to this function - this is the corrected V2 from last time)
    """(REFACTORED handles conflict only in this method to merge in main) Fits an outlier detector on X..."""
    if configs['outlier_detection'] == OUTLIER_DETECTORS[0]:  # z-score (STATEFUL, MEAN-BASED)
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


def xgb_feature_importance_selector(X, y, importance_thresh=0.0001):
    # (No changes to this function)
    """Fits an XGBoost model and selects features based on importance threshold..."""
    model = XGBRegressor(random_state=configs["random_state"], n_estimators=100, verbosity=0)
    model.fit(X, y)
    importances = model.feature_importances_
    keep_mask = importances > importance_thresh
    return keep_mask, importances


class FeatureSelector:
    # (No changes to this class)
    """Feature selection to remove irrelevant or redundant features..."""

    def __init__(self, var_thresh=None, corr_thresh=None, xgb_thresh=None, print_removed_ones=False):
        self.var_thresh = var_thresh
        self.corr_thresh = corr_thresh
        self.xgb_thresh = xgb_thresh
        self.mask_ = None
        self.print_removed_ones = print_removed_ones

    def fit(self, X, y):
        # (Omitted for brevity - no changes)
        df_original = pd.DataFrame(X)
        df_filtered = df_original.copy()
        if self.var_thresh is not None:
            df_for_var = df_filtered.copy()
            col_means = df_for_var.mean(numeric_only=True, skipna=True)
            df_for_var = df_for_var.fillna(col_means)
            selector = VarianceThreshold(threshold=self.var_thresh)
            var_mask = selector.fit(df_for_var).get_support()
            n_removed_var = np.sum(~var_mask)
            print(f"Variance filter (thresh={self.var_thresh}): Removed {n_removed_var} features")
            if n_removed_var > 0 and self.print_removed_ones:
                dropped_mask_var = ~var_mask
                dropped_cols_var = df_original.columns[dropped_mask_var]
                dropped_vars = selector.variances_[dropped_mask_var]
                for col, var_val in zip(dropped_cols_var, dropped_vars):
                    print(f"  - Column {col}: variance = {var_val:.6f}")
            df_filtered = df_filtered.iloc[:, var_mask]
        if self.corr_thresh is not None:
            corr_matrix = df_filtered.corr(numeric_only=True).abs()
            upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            highly_correlated_features_to_drop = [col for col in upper_triangle.columns if
                                                  any(upper_triangle[col] > self.corr_thresh)]
            corr_mask = ~df_filtered.columns.isin(highly_correlated_features_to_drop)
            n_removed_corr = len(highly_correlated_features_to_drop)
            print(f"Correlation filter (thresh={self.corr_thresh}): Removed {n_removed_corr} features")
            if n_removed_corr > 0 and self.print_removed_ones:
                for feat in highly_correlated_features_to_drop:
                    max_corr_col = upper_triangle[feat].idxmax()
                    max_corr_val = upper_triangle[feat].max()
                    print(f"  - Column {feat}: max correlation = {max_corr_val:.4f} (with Column {max_corr_col})")
            df_filtered = df_filtered.loc[:, corr_mask]
        if self.xgb_thresh is not None:
            X_for_xgb = df_filtered.values
            xgb_mask, importances = xgb_feature_importance_selector(X_for_xgb, y, importance_thresh=self.xgb_thresh)
            dropped_mask_xgb = ~xgb_mask
            n_removed_xgb = np.sum(dropped_mask_xgb)
            print(f"XGBoost filter (thresh={self.xgb_thresh}): Removed {n_removed_xgb} features")
            if n_removed_xgb > 0 and self.print_removed_ones:
                dropped_cols_xgb = df_filtered.columns[dropped_mask_xgb]
                dropped_importances = importances[dropped_mask_xgb]
                for col, imp_val in zip(dropped_cols_xgb, dropped_importances):
                    print(f"  - Column {col}: importance = {imp_val:.6f}")
            df_filtered = df_filtered.iloc[:, xgb_mask]
        self.mask_ = df_original.columns.isin(df_filtered.columns)
        print(f"✅ Selected {self.mask_.sum()} / {X.shape[1]} features "
              f"({self.mask_.sum() / X.shape[1] * 100:.1f}%)")
        return self

    def transform(self, X):
        # (No changes to this method)
        if self.mask_ is None:
            raise ValueError("Must call fit before transform.")
        return X[:, self.mask_]


# === MODIFIED: fit function ===
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
    return model


def train_model(X, y, i=None):
    # (No changes to this function)
    """(REFACTORED, DO NOT MERGE IN MAIN) Run training pipeline..."""
    imputer = imputation(X, i)
    X_imputed = imputer.transform(X)

    get_keep_mask = outlier_detection(X_imputed, y)

    train_mask = get_keep_mask(X_imputed)
    X_filtered = X_imputed[train_mask, :]
    y_filtered = y[train_mask]

    print(f"Outlier detection: Kept {X_filtered.shape[0]} / {X_imputed.shape[0]} samples")

    selection = FeatureSelector(var_thresh=configs["var_thresh"], corr_thresh=configs["corr_thresh"],
                                xgb_thresh=configs["xgb_thresh"], print_removed_ones=configs["print_removed_ones"])
    selection.fit(X_filtered, y_filtered)
    X_filtered_selected = selection.transform(X_filtered)

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

                    # Pipeline (no change)
                    imputer, get_keep_mask, selection, model, x_train_final, y_train_final = train_model(
                        x_train, y_train, i
                    )

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

        imputer, _, selection, model, _, _ = train_model(x_train, y_train)

        x_test_imputed = imputer.transform(x_test)
        x_test_selected = selection.transform(x_test_imputed)
        y_test_pred = model.predict(x_test_selected)

        table = pd.DataFrame(
            {"id": np.arange(0, y_test_pred.shape[0]), "y": y_test_pred.flatten()}
        )
        table.to_csv("./submission.csv", index=False)
        print("\n✅ Successfully generated submission.csv")