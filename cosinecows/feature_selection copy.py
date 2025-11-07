"""
Generate features for modeling.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, mutual_info_regression, SelectFromModel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cosinecows.config import configs


class CorrelationRemover(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.cols_to_keep_ = None  # Stores *integer indices*

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > self.threshold)]
        self.cols_to_keep_ = [i for i in range(X.shape[1]) if i not in to_drop]
        return self

    def transform(self, X):
        return X[:, self.cols_to_keep_]


class PassthroughSelector(BaseEstimator, TransformerMixin):
    """
    hacky simple transformer passing data through without modification.
    Used to bypass feature selection.
    """
    def fit(self, X, y=None):
        return self # Nothing to fit

    def transform(self, X):
        return X # Just return the data


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


def make_feature_selection_pipeline(
    thresh_var=0.01,
    thresh_corr=0.95,
    rf_max_feats=120,
    rf_n_estimators=70,
    percentile=40,
    random_state=42
):
    rf = RandomForestRegressor(
        n_estimators=rf_n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    rf_selector = SelectFromModel(
        rf, max_features=rf_max_feats, threshold='0.1*mean'
    )

    return make_pipeline(
        VarianceThreshold(threshold=thresh_var),
        PrintShape(message="after VarianceThreshold"),
        CorrelationRemover(threshold=thresh_corr),
        PrintShape(message="after CorrelationRemover"),
        SelectPercentile(score_func=mutual_info_regression, percentile=percentile),
        PrintShape(message="after SelectPercentile"),
        rf_selector,
        PrintShape(message="after RF SelectFromModel")
    )
