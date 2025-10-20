import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

if __name__ == "__main__":
    x_train_df = pd.read_csv('./data/X_train.csv', skiprows=1, header=None)
    y_train_df = pd.read_csv('./data/y_train.csv', skiprows=1, header=None)
    x_test_df = pd.read_csv('./data/X_test.csv', skiprows=1, header=None)

    x_train = x_train_df.values[:, 1:]
    y_train = y_train_df.values[:, 1:]
    x_test = x_test_df.values[:, 1:]

    # Randomly split the data into training and validation sets with 80-20 ratio
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    # Impute missing values with mean of each column
    x_mean = np.nanmean(x_train, axis=0, keepdims=True)
    x_train = np.where(np.isnan(x_train), x_mean, x_train)
    x_val = np.where(np.isnan(x_val), x_mean, x_val)
    x_test = np.where(np.isnan(x_test), x_mean, x_test)

    # Select top 100 features with highest mutual information
    selection = SelectKBest(mutual_info_regression, k=100).fit(x_train, y_train)
    x_train = selection.transform(x_train)
    x_val = selection.transform(x_val)
    x_test = selection.transform(x_test)

    # Train a linear regression model
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)

    y_train_pred = regressor.predict(x_train)
    y_val_pred = regressor.predict(x_val)

    # Evaluate the model on training and validation sets
    train_score = r2_score(y_train, y_train_pred)
    val_score = r2_score(y_val, y_val_pred)

    print(train_score, val_score)

    # Predict on test set
    y_test_pred = regressor.predict(x_test)

    # Save predictions to submission file with the given format
    table = pd.DataFrame({'id': np.arange(0, y_test_pred.shape[0]), 'y': y_test_pred.flatten()})
    table.to_csv('./submission.csv', index=False)