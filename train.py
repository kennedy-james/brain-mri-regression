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
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.ensemble import StackingRegressor
from sklearn.svm import SVR

class RunMode(Enum):
    final_evaluation = auto() # produce submission file for test data
    wandb = auto() # log to wandb
    grid = auto()   # run all combinations of models and outlier detectors locally
    current_config = auto() # run single CV with current config


class Imputer(Enum):
    mean = auto()
    median = auto()
    most_frequent = auto()
    knn = auto()
    iterative = auto()


class OutlierDetector(Enum):
    zscore = auto()
    knn = auto()
    isoforest = auto()
    svm = auto()
    pca_svm = auto()
    pca_isoforest = auto()



class Regressor(Enum):
    xgb = auto()
    extra_trees = auto()
    ridge = auto()
    random_forest_regressor = auto()
    stacking = auto()


RUN_MODE = RunMode.current_config

configs = {
    'folds': 10,
    'random_state': 42,
    'impute_method': Imputer.knn,
    'knn_neighbours': 75,
    'knn_weight': 'uniform',  # possible neighbour weights for average (uniform, distance)
    'iterative_estimator': 'Ridge()',  # Iterative configuration
    'iterative_iter': 1,  # Iterative configuration
    'outlier_detector': {
        'method': OutlierDetector.pca_isoforest,
        'zscore_std': 1,
        'pca_n_components': 2,
        'pca_svm_nu': 0.05,    # "expected amount of outliers to discard"
        'pca_svm_gamma': 0.0003, # blurriness of internal holes within clusters
        'pca_isoforest_contamination': 0.045, # proportion of outliers
    },
    'regression_method': Regressor.stacking,
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
    if method in [Imputer.mean, Imputer.median, Imputer.most_frequent]:
        imputer = SimpleImputer(strategy=configs["impute_method"].name)
        imputer.fit(X)
    elif method is Imputer.knn:
        scaler = StandardScaler()
        knn_imputer = KNNImputer(n_neighbors=configs["knn_neighbours"], weights=configs["knn_weight"])
        imputer = pipeline.make_pipeline(scaler, knn_imputer)
        imputer.fit(X)
    elif method is Imputer.iterative: # iterative imputer
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
            num_total = X_data.shape[0]
            num_inliers = np.sum(detector)
            num_outliers = num_total - num_inliers

            if num_inliers == 0:
                print(f"WARNING: {method} removed all samples. Keeping all as fallback.")
                return np.ones(X_data.shape[0], dtype=bool)

            if X_data.shape[0] == X.shape[0]:
                print(f"INFO: {method} (on train data) found {num_outliers} outliers. Keeping {num_inliers} / {num_total} samples.")
            return detector

        return wrapped

    method = configs['outlier_detector']['method']
    if method is OutlierDetector.zscore:
        threshold = configs['outlier_detector']['zscore_std'] # std devs
        print(f"Using z-score detector (stateful, mean-based, threshold={threshold})")
        mean_train = np.nanmean(X, axis=0)
        std_train = np.nanstd(X, axis=0)
        std_train[std_train == 0] = 1.0

        def predict_fn(X_data):
            zscores = np.abs((X_data - mean_train) / std_train)
            return np.mean(zscores, axis=1) <= threshold

        get_detector = safe_detector(predict_fn)

    elif method is OutlierDetector.knn:
        clf = KNN(contamination=0.05)
        clf.fit(X)
        print(f"Using KNN detector (stateful, contamination={clf.contamination})")
        get_detector = safe_detector(lambda X_data: clf.predict(X_data) == 0) # inliers are labeled 0, outliers 1

    elif method is OutlierDetector.isoforest:
        iso = IsolationForest(contamination=0.05, random_state=configs["random_state"])
        iso.fit(X)
        print(f"Using IsolationForest (stateful, contamination={iso.contamination})")
        get_detector = safe_detector(lambda X_data: iso.predict(X_data) == 1) # inliers are labeled 1, outliers -1

    elif method is OutlierDetector.svm:  # One-Class SVM
        # nu ~ contamination, upper bound on fraction of training errors and lower bound of fraction of support vectors.
        clf = OneClassSVM(nu=0.05, kernel='rbf', gamma='scale') # nu in [0, 0.5]
        clf.fit(X)
        print(f"Using One-Class SVM (stateful, nu={clf.nu})")
        get_detector = safe_detector(lambda X_data: clf.predict(X_data) == 1) # inliers 1, outliers -1

    elif method is OutlierDetector.pca_svm:
        # scale data, pca, svm on low-dim representation
        n_components_pca = 2  # hyperparam
        pca_svm_pipeline = make_pipeline(
            StandardScaler(),
            PCA(n_components=configs['outlier_detector']['pca_n_components'], random_state=configs["random_state"]),
            OneClassSVM(nu=configs['outlier_detector']['pca_svm_nu'], kernel='rbf', gamma=configs['outlier_detector']['pca_svm_gamma'])  # rbf is fast on low-dim data
        )

        print(f"Using PCA+SVM detector (stateful, n_components={n_components_pca}, nu={configs['outlier_detector']['pca_svm_nu']}, gamma={configs['outlier_detector']['pca_svm_gamma']})")
        pca_svm_pipeline.fit(X)
        get_detector = safe_detector(lambda X_data: pca_svm_pipeline.predict(X_data) == 1) # inliers 1, outliers -1

    elif method is OutlierDetector.pca_isoforest:  # PCA + Isolation Forest
        # IsoForest is not sensitive to scale, so StandardScaler isn't  strictly required for the model, but for PCA.
        pca_isoforest_pipeline = make_pipeline(
            StandardScaler(),
            PCA(n_components=configs['outlier_detector']['pca_n_components'], random_state=configs["random_state"]),
            IsolationForest(contamination=configs['outlier_detector']['pca_isoforest_contamination'], random_state=configs["random_state"])
        )
        print(f"Using PCA+IsolationForest detector (stateful, n_components={configs['outlier_detector']['pca_n_components']}, contamination={configs['outlier_detector']['pca_isoforest_contamination']})")
        pca_isoforest_pipeline.fit(X)
        get_detector = safe_detector(lambda X_data: pca_isoforest_pipeline.predict(X_data) == 1) # inliers 1, outliers -1

    else:
        raise ValueError(f"Unknown outlier detection method: {method}")

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
    
class PrintShape(BaseEstimator, TransformerMixin):
    def __init__(self, message=""):
        self.message = message

    def fit(self, X, y=None):
        # No-op: just return self (required for fitting)
        return self

    def transform(self, X):
        # Print the shape (focus on n_features = X.shape[1])
        print(f"number of remaining features {self.message}: {X.shape[1]}")
        return X  # Pass through unchanged


def feature_selection(x_train, y_train, thresh_var=0.01, thresh_corr=0.93, rf_max_feats=250, rf_n_estimators=70):
    if hasattr(x_train, 'values'):
        x_train = x_train.values
        print("Converted to numpy array")

    rf = RandomForestRegressor(n_estimators=rf_n_estimators, random_state=configs['random_state'], n_jobs=-1)
    rf_selector = SelectFromModel(rf, max_features=rf_max_feats, threshold='0.1*mean')

    selection = make_pipeline(
        VarianceThreshold(threshold=thresh_var),  # low variance removal
        PrintShape(message="after VarianceThreshold"),  # Logs after this step
        CorrelationRemover(threshold=thresh_corr),  # high correlation removal
        PrintShape(message="after CorrelationRemover"),  # Logs after this step
        SelectPercentile(score_func=mutual_info_regression, percentile=40),  # equivalent to KBest=200, more robust
        PrintShape(message="after SelectPercentile"),  # Logs after this step
        rf_selector  # non-linear embedded selection (RF instead of Lasso)
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

    if model_name is Regressor.xgb:
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
            )),
            ('ridge', make_pipeline(
                StandardScaler(),
                Ridge(random_state=configs["random_state"])
            )),
            ('svr_linear', make_pipeline(
                StandardScaler(),
                SVR(kernel='linear', C=0.1) # rbf is too slow
            ))
        ]

        # meta-model combining predictions: simple robust Ridge model.
        final_estimator = Ridge(random_state=configs["random_state"])

        # cv=5 means it will use 5-fold cross-validation internally to generate predictions, which prevents data leakage.
        model = StackingRegressor(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5,  # Use 5 folds, 10 is too slow
            n_jobs=-1  # Use all cores
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
    outlier_method = configs["outlier_detector"]['method']
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
        y_train_pred = model.predict(x_proc)
        train_score = r2_score(y_proc, y_train_pred)

        # Validation Pipeline
        x_val_imputed = imputer.transform(x_val)
        x_val_selected = selection.transform(x_val_imputed) # apply selection (NO outlier removal on validation data)
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


def log_results_to_wandb(cv_df, run):
    """Logs a CV result DataFrame to a wandb run.

    Parameters:
    ----------
    cv_df: DataFrame with CV results
    run: W&B run object

    Returns:
    -----------
    None
    """
    print("\nLogging results to W&B...")
    # Generate and log boxplot
    fig, ax = plt.subplots(figsize=(11, 13))
    sns.boxplot(data=cv_df[["train_score", "validation_score"]], ax=ax)
    ax.set_title("Cross-Validation Results")
    ax.set_ylabel("R² Score")
    ax.set_xlabel("Score Type")
    run.log({"CV_Boxplot": wandb.Image(fig)})
    plt.close(fig)
    # Store raw CV results in table
    cv_table = wandb.Table(dataframe=cv_df)
    run.log({"CV Results": cv_table})
    # Log summary statistics
    run.summary["mean_train_score"] = cv_df["train_score"].mean()
    run.summary["mean_validation_score"] = cv_df["validation_score"].mean()
    run.summary["std_train_score"] = cv_df["train_score"].std()
    run.summary["std_validation_score"] = cv_df["validation_score"].std()
    print("✅ W&B run complete.")


def save_results_locally(results_df, is_grouped_run):
    """Saves a results DataFrame locally to CSV and creates a boxplot.

    Parameters:
    ----------
    results_df: DataFrame with CV results
    is_grouped_run: Boolean indicating if multiple model/outlier combinations are included.

    Returns:
    -----------
    None
    """
    print("\n\n--- 📊 Final Performance Summary ---")

    # save csv
    csv_filename = "cv_run_results.csv"
    if is_grouped_run:
        csv_filename = "cv_run_results_all.csv"

    results_df.to_csv(csv_filename, index=False)
    print(f"\n✅ All results saved to '{csv_filename}'")

    print("\n--- Validation R² Summary ---")
    if is_grouped_run:
        summary_stats = results_df.groupby(['model', 'outlier_method'])['validation_score'].describe()
    else:
        summary_stats = results_df['validation_score'].describe()
    print(summary_stats)

    # generate boxplot
    fig, ax = plt.subplots(figsize=(14, 8))
    plot_filename = "cv_run_boxplot.png"

    if is_grouped_run:
        sns.boxplot(data=results_df, x='outlier_method', y='validation_score', hue='model', ax=ax)
        ax.set_title("Model Comparison by Outlier Method (Validation R²)")
        ax.set_xlabel("Outlier Detection Method")
        ax.legend(title="Model")
        plot_filename = "cv_run_boxplot_all.png"
    else:
        sns.boxplot(data=results_df[["train_score", "validation_score"]], ax=ax)
        ax.set_title(f"CV Results: {configs['regression_method'].name} + {configs['outlier_detector']['method'].name}")
        ax.set_xlabel("Score Type")

    ax.set_ylabel("R² Score")
    plt.savefig(plot_filename)
    print(f"\n✅ Saved boxplot to '{plot_filename}'")
    print("\n✅ Local run complete.")


if __name__ == "__main__":
    x_training_data = pd.read_csv("./data/X_train.csv", skiprows=1, header=None).values[:, 1:]
    y_training_data = (pd.read_csv("./data/y_train.csv", skiprows=1, header=None).values[:, 1:].ravel())

    if RUN_MODE == RunMode.final_evaluation:
        # Generates submission.csv using the single configuration defined in the global 'configs' dict
        print(f"🚀 Running final evaluation pipeline with:")
        print(f"   Model: {configs['regression_method'].name}")
        print(f"   Outlier Detector: {configs['outlier_detection'].name}")

        x_test = pd.read_csv("./data/X_test.csv", skiprows=1, header=None).values[:, 1:]
        x_train = x_training_data
        y_train = y_training_data

        # Pipeline to fit on training set
        imputer, detector, selection, model, _, _ = train_model(x_train, y_train)
        x_test_imputed = imputer.transform(x_test)
        x_test_selected = selection.transform(x_test_imputed) # apply feature selection, NO outlier removal
        y_test_pred = model.predict(x_test_selected)

        # Save predictions to submission file
        table = pd.DataFrame({"id": np.arange(0, y_test_pred.shape[0]), "y": y_test_pred.flatten()})
        table.to_csv("./submission.csv", index=False)
        print("\n✅ Successfully generated submission.csv")

    elif RUN_MODE == RunMode.wandb:
        # Runs a single CV experiment (using 'configs') and logs to W&B.
        print(f"🚀 Starting W&B run for: {configs['regression_method']} + {configs['outlier_detection']}")
        with wandb.init(
            project="AML_task1",
            config=configs,
            tags=["regression", configs["regression_method"].name, configs["outlier_detection"].name],
            name=f"regressor {configs['regression_method'].name}_{configs['outlier_detection'].name}",
            notes=f''
        ) as run:
            cv_df = run_cv_experiment(x_training_data, y_training_data)
            log_results_to_wandb(cv_df, run)

    elif RUN_MODE == RunMode.current_config:
        print(f"🚀 Starting single local CV run for: {configs['regression_method'].name} + {configs['outlier_detector']['method'].name}")
        cv_df = run_cv_experiment(x_training_data, y_training_data)
        save_results_locally(cv_df, is_grouped_run=False)  # Use the helper

    elif RUN_MODE == RunMode.grid:
        # Runs all combinations of models and outlier detectors locally. Saves one CSV and one plot with all results.
        print("🚀 Starting local 'Run All' comparison...")
        all_results_dfs = []

        for model_name in Regressor:
            configs["regression_method"] = model_name  # !update global config

            # for outlier_method in OutlierDetector:
            # configs["outlier_detection"] = outlier_method  # !update global config
            cv_df = run_cv_experiment(x_training_data, y_training_data)
            all_results_dfs.append(cv_df)

        results_df = pd.concat(all_results_dfs)
        save_results_locally(results_df, is_grouped_run=True)