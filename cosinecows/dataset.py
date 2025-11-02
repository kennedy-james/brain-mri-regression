"""
Loads datasets.
"""
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
INTERIM_DATA_DIR = DATA_DIR / 'interim'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
REPORTS_DIR = PROJECT_ROOT / 'reports'
FIGURES_DIR = REPORTS_DIR / 'figures'
MODELS_DIR = PROJECT_ROOT / 'models'
IMPUTERS_DIR = MODELS_DIR / 'imputers'

def load_train_data():
    """Loads training data from CSV files.

    Returns:
    ----------
    x: Features array.
    y: Labels array.
    """
    x = pd.read_csv(RAW_DATA_DIR / 'X_train.csv', skiprows=1, header=None).values[:, 1:]
    y = (pd.read_csv(RAW_DATA_DIR / 'y_train.csv', skiprows=1, header=None).values[:, 1:].ravel())
    return x, y


def load_test_data():
    """Loads test data from CSV file.

    Returns:
    ----------
    x_test: Test features array.
    """
    x_test = pd.read_csv(RAW_DATA_DIR / 'X_test.csv', skiprows=1, header=None).values[:, 1:]
    return x_test
