import os

import joblib
from sklearn import pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cosinecows.config import configs, Imputer
from cosinecows.dataset import IMPUTERS_DIR


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
        loadable_file = IMPUTERS_DIR / f'{configs["iterative_estimator"].split("(")[0]}{configs["iterative_iter"]}_{i}.pkl'
        if i is not None and os.path.isfile(loadable_file):
            print(f"--- 🔄 Loading existing imputer from {loadable_file} ---")
            imputer = joblib.load(loadable_file)
        else:
            print(f"--- 🛠️ Training new imputer and saving to {loadable_file} ---")
            if configs["iterative_estimator"] == 'Ridge()':
                imputer = IterativeImputer(
                    random_state=configs["random_state"],
                    estimator=make_pipeline(
                        StandardScaler(),
                        Ridge()
                    ),
                    max_iter=configs["iterative_iter"],
                )
                imputer.fit(X)
                joblib.dump(imputer, loadable_file)
    return imputer


