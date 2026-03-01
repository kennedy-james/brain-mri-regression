"""
Generate features for modeling.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, VarianceThreshold, SelectPercentile, mutual_info_regression, f_regression, SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import Lasso

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


def feature_selection_old(thresh_var=0.01, k_best=200):


    print(f'Using feature selection pipeline with: {thresh_var = }, {k_best = }')
    selection = make_pipeline(
        SimpleImputer(strategy='median'),  # ensure no NaNs
        RobustScaler(),  # robust scaling
        VarianceThreshold(threshold=thresh_var),  # low variance removal
        PrintShape(message="after VarianceThreshold"),  # Logs after this step
        SelectKBest(score_func=f_regression, k=k_best),
    )
    return selection


def feature_selection(thresh_var=0.01, score_func='f_regression', k_best=200):
    def rf(X, y):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model.feature_importances_, None

    def lasso(X, y):
        model = Lasso(alpha=0.1)
        model.fit(X, y)
        return np.abs(model.coef_), None

    # Build pipeline steps dynamically
    steps = [
        PrintShape(message="before VarianceThreshold"),
        RobustScaler(),
        VarianceThreshold(threshold=thresh_var),  # low variance removal
        PrintShape(message="after VarianceThreshold"),  # Logs after this step
    ]

    # Map string to function after deciding on scaler
    if score_func in ['f_regression', 'lasso_regression']:
        steps.append(StandardScaler())

    if score_func == 'f_regression':
        score_func = f_regression
    elif score_func == 'mutual_info_regression':
        score_func = mutual_info_regression
    elif score_func == 'random_forest_regressor':
        score_func = rf
    elif score_func == 'lasso_regression':
        score_func = lasso
    else:
        raise ValueError(f"Unknown score_func: {score_func}")

    steps.extend([
        SelectKBest(score_func=score_func, k=k_best),
        PrintShape(message="after SelectKBest"),
        # CorrelationRemover(threshold=thresh_corr),  # high correlation removal
        # PrintShape(message="after CorrelationRemover"),  # Logs after this step
        # SelectPercentile(score_func=mutual_info_regression, percentile=percentile),  # Use the passed percentile
        # PrintShape(message="after SelectPercentile"),  # Logs after this step
        # rf_selector,  # non-linear embedded selection (RF instead of Lasso)
        # PrintShape(message="after RandomForestSelector")  # Logs after this step
    ])

    selection = make_pipeline(*steps)
    return selection