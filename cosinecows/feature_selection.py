"""
Generate features for modeling.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, mutual_info_regression
from sklearn.pipeline import make_pipeline


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


def feature_selection(x_train, y_train, thresh_var=0.01, thresh_corr=0.95, rf_max_feats=120, rf_n_estimators=70, percentile=40):
    if hasattr(x_train, 'values'):
        x_train = x_train.values
        print("Converted to numpy array")

    # rf = RandomForestRegressor(n_estimators=rf_n_estimators, random_state=configs['random_state'], n_jobs=-1)
    # rf_selector = SelectFromModel(rf, max_features=rf_max_feats, threshold='0.1*mean')
    univariate_selector = SelectPercentile(
        score_func=mutual_info_regression,
        percentile=percentile
    )

    selection = make_pipeline(
        VarianceThreshold(threshold=thresh_var),  # low variance removal
        CorrelationRemover(threshold=thresh_corr),  # high correlation removal
        # SelectPercentile(score_func=mutual_info_regression, percentile=30), # equivalent to KBest=200, more robust
        # rf_selector # non-linear embedded selection (RF instead of Lasso)
        univariate_selector # selects top x% feats based on mutual info
    )
    selection.fit(x_train, y_train)
    return selection
