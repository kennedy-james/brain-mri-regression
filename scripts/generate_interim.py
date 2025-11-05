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


INTERIM_MODE = InterimMode.imputation

IMP_FILENAME = f'X_interim_{configs['impute_method'].name}'
match configs['impute_method']:
    case Imputer.knn:
        IMP_FILENAME += f'_{configs["knn_neighbours"]}n_{configs["knn_weight"]}w'
    case Imputer.iterative:
        IMP_FILENAME += f'_{configs['iterative_estimator'][:-2]}_{configs['iterative_iter']}iter'
    case _:
        raise NotImplementedError(
            f"Imputation {configs['impute_method'].name} not implemented for interim data generation.")


def generate_imputation(x):
    """ Uses imputation method set in configs.py """
    global IMP_FILENAME
    print(f"Generating interim data from imputer {configs['impute_method'].name}...")
    x, _ = load_train_data()
    imputer = imputation(x, i=None)
    x_imp = imputer.transform(x)
    filepath = INTERIM_DATA_DIR / IMP_FILENAME
    interim_file = filepath.with_suffix('.csv')
    print(f"Saving imputed data to {interim_file}...")
    pd.DataFrame(x_imp).to_csv(interim_file, index=False, header=False)


def generate_outlier(x):
    """ Uses outlier method set in configs.py """
    print(f"Generating outlier {configs['outlier_method'].name} data from configured imputer {configs['impute_method'].name}...")
    x, _ = load_train_data()
    imputer = imputation(x, i=None)
    x_imp = imputer.transform(x)
    filepath = INTERIM_DATA_DIR / filename
    interim_file = filepath.with_suffix('.csv')
    print(f"Saving imputed data to {interim_file}...")
    pd.DataFrame(x_imp).to_csv(interim_file, index=False, header=False)


def generate_feature_selection(x):
    """ Uses feature_selection method set in configs.py """
    print(f"Generating feature selected data from configured outlier {configs['outlier_method'].name} and {configs['impute_method'].name}...")
    pass

def main():


if __name__ == "__main__":
    match INTERIM_MODE:
        case InterimMode.imputation: generate_imputation()
        case InterimMode.outlier: generate_outlier()
        case InterimMode.feature_selection: generate_feature_selection()
