import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer

# Set to 'True' to produce submission file for test data
FINAL_EVALUATION = False

# Reproducible dictionary defining experiment
configs = {"folds": 10, "impute_method": "mean", "random_state": 42}


def imputation(X):
    """Replace missing values in dataset using imputation

    Parameters
    ----------
    X: Dataset to learn imputation rule

    Returns
    ----------
    imputer: Trained imputer for imputing new data points
    """
    # TODO: Implement effective imputation
    imputer = SimpleImputer(missing_values=np.nan, strategy=configs["impute_method"])
    imputer.fit(X)

    return imputer


def outlier_detection(X, y):
    """Detect outlier data points that are to be removed

    Parameters
    ----------
    X: Features on which to train outlier prediction
    y: Output for associated features

    Returns
    ----------
    detector: Detector that returns indices of outliers that should be deleted
    """

    # TODO: Replace detector with one that returns indices that are supposed to be deleted
    detector = lambda x: np.array([])
    return detector


def feature_selection(X, y):
    """Train feature selector that removes redundant features

    Parameters
    ----------
    X: Features on which to train feature selector
    y: Output for associated features

    Returns
    ----------
    selector: Trained feature selector that can be applied to other data points
    """
    # TODO: Implement effective feature selector
    selector = SelectKBest(mutual_info_regression, k=100).fit(X, y)

    return selector


def fit(X, y):
    """Training of the model

    Parameters
    ----------
    X: Training data
    y: Output to learn correct prediction

    Returns
    ----------
    model: Final model for prediction
    """
    # TODO: Implement effective regression model
    model = LinearRegression()
    model.fit(X, y)

    return model

def train_model(X, y):
    """Run training pipeline 
    
    Parameters
    ----------
    X: Training data
    y: Output to learn correct prediction

    Returns
    ----------
    imputer: Trained imputation model
    detector: Trained detector model
    selection: Trained selection model
    model: Trained prediction model
    X: Manipulated training data
    y: Manipulated training labels
    """
    imputer = imputation(X)
    X = imputer.transform(X)

    detector = outlier_detection(X, y)
    outlier_indices = detector(X)
    X = np.delete(X, outlier_indices, axis=0)
    y = np.delete(y, outlier_indices)

    selection = feature_selection(X, y)
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
        # Apply KFold CV for model selection
        folds = KFold(n_splits=configs["folds"])
        for train_index, validation_index in folds.split(x_training_data):
            x_val = x_training_data[validation_index, :]
            y_val = y_training_data[validation_index]
            x_train = x_training_data[train_index, :]
            y_train = y_training_data[train_index]

            # Pipeline to fit on training set
            imputer, detector, selection, model, x_train = train_model(x_train, y_train)
            y_train_pred = model.predict(x_train)

            # Pipeline to perform predictions on validation set
            x_val = imputer.transform(x_val)
            outlier_indices = detector(x_val)
            x_val = np.delete(x_val, outlier_indices, axis=0)
            y_val = np.delete(y_val, outlier_indices)
            x_val = selection.transform(x_val)

            y_val_pred = model.predict(x_val)

            # Evaluate the model on training and validation sets
            train_score = r2_score(y_train, y_train_pred)
            val_score = r2_score(y_val, y_val_pred)

            print(train_score, val_score)
    else:
        x_test = pd.read_csv("./data/X_test.csv", skiprows=1, header=None).values[:, 1:]
        x_train = x_training_data
        y_train = y_training_data

        # Pipeline to fit on training set
        imputer, detector, selection, model, _ = train_model(x_train, y_train)

        # Pipeline to perform predictions on test set
        x_test = imputer.transform(x_test)
        outlier_indices = detector(x_test)
        x_test = np.delete(x_test, outlier_indices, axis=0)
        x_test = selection.transform(x_test)

        y_test_pred = model.predict(x_test)

        # Save predictions to submission file with the given format
        table = pd.DataFrame(
            {"id": np.arange(0, y_test_pred.shape[0]), "y": y_test_pred.flatten()}
        )
        table.to_csv("./submission.csv", index=False)
