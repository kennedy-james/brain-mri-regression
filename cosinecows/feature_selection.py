"""
Generate features for modeling.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, mutual_info_regression, SelectFromModel, \
    SelectKBest
from sklearn.linear_model import Lasso, LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

from cosinecows.config import configs

from cosinecows.config import configs


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


def feature_selection(x_train, y_train, thresh_var, thresh_corr, percentile):
    if hasattr(x_train, 'values'):
        x_train = x_train.values
        print("Converted to numpy array")

    # rf = RandomForestRegressor(n_estimators=rf_n_estimators, random_state=configs['random_state'], n_jobs=-1)
    # rf_selector = SelectFromModel(rf, max_features=rf_max_feats, threshold='0.1*mean') params rf_max_feats, rf_n_estimators,

    selection = make_pipeline(
        StandardScaler(),
        VarianceThreshold(threshold=thresh_var),  # low variance removal
        PrintShape(message="after VarianceThreshold"),  # Logs after this step
        CorrelationRemover(threshold=thresh_corr),  # high correlation removal
        PrintShape(message="after CorrelationRemover"),  # Logs after this step
        # SelectPercentile(score_func=mutual_info_regression, percentile=percentile),  # equivalent to KBest=200, more robust
        SelectKBest(score_func=mutual_info_regression, k=200),  # k best
        SelectFromModel(LassoCV(n_jobs=-1, random_state=configs['random_state'])),  # linear embedded selection (Lasso)
        # PrintShape(message="after SelectPercentile"),  # Logs after this step
        # rf_selector  # non-linear embedded selection (RF instead of Lasso)
        # PrintShape(message="after random forest"),  # Logs after this step
    )
    selection.fit(x_train, y_train)
    return selection
