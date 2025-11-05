from cosinecows.config import configs, Imputer
from cosinecows.imputation import imputation
from cosinecows.dataset import load_train_data
from enum import Enum, auto
import pandas as pd
from cosinecows.dataset import INTERIM_DATA_DIR

class InterimMode(Enum):
    """Sequential steps need previous step data to proceed."""
    imputation = auto()
    outlier = auto()
    feature_selection = auto()


INTERIMMODE = InterimMode.imputation


def generate_imputation():
    """ Uses imputation method set in configs.py """
    print("Generating interim data...")
    x, _ = load_train_data()
    imputer = imputation(x, i=None)
    x_imp = imputer.transform(x)
    filename = f'X_interim_{configs['impute_method'].name}'

    if configs['impute_method'] == Imputer.knn:
        filename += f'_{configs["knn_neighbours"]}n_{configs["knn_weight"]}w'
    elif configs['impute_method'] == Imputer.iterative:
        filename += f'_{configs['iterative_estimator'][:-2]}_{configs['iterative_iter']}iter'

    filepath = INTERIM_DATA_DIR / filename
    interim_file = filepath.with_suffix('.csv')
    print(f"Saving imputed data to {interim_file}...")
    pd.DataFrame(x_imp).to_csv(interim_file, index=False, header=False)


def generate_outlier():
    pass


def generate_feature_selection():
    pass


if __name__ == "__main__":
    match INTERIMMODE:
        case InterimMode.imputation: generate_imputation()
        case InterimMode.outlier: generate_outlier()
        case InterimMode.feature_selection: generate_feature_selection()
