import numpy as np
from pyod.models.knn import KNN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import OneClassSVM

from cosinecows.config import configs, OutlierDetector


def outlier_detection(X, y):
    """Detect outlier data points / samples that are to be removed

    Parameters
    ----------
    X: Features on which to train outlier prediction
    y: Labels for associated features

    Returns
    ----------
    detector: Detector that returns indices of inliers that should be kept
    """
    def safe_detector(predict_fn):
        def wrapped(X_data):
            detector = predict_fn(X_data)
            num_total = X_data.shape[0]
            num_inliers = np.sum(detector)
            num_outliers = num_total - num_inliers

            if num_inliers == 0:
                print(f"WARNING: {method} removed all samples. Keeping all as fallback.")
                return np.ones(X_data.shape[0], dtype=bool)

            if X_data.shape[0] == X.shape[0]:
                print(f"INFO: {method} (on train data) found {num_outliers} outliers. Keeping {num_inliers} / {num_total} samples.")
            return detector

        return wrapped

    method = configs['outlier_method']
    if method is OutlierDetector.zscore:
        threshold = configs['zscore_std'] # std devs
        print(f"Using z-score detector (stateful, mean-based, threshold={threshold})")
        mean_train = np.nanmean(X, axis=0)
        std_train = np.nanstd(X, axis=0)
        std_train[std_train == 0] = 1.0

        def predict_fn(X_data):
            zscores = np.abs((X_data - mean_train) / std_train)
            return np.mean(zscores, axis=1) <= threshold

        get_detector = safe_detector(predict_fn)

    elif method is OutlierDetector.knn:
        clf = KNN(contamination=0.05)
        clf.fit(X)
        print(f"Using KNN detector (stateful, contamination={clf.contamination})")
        get_detector = safe_detector(lambda X_data: clf.predict(X_data) == 0) # inliers are labeled 0, outliers 1

    elif method is OutlierDetector.isoforest:
        iso = IsolationForest(contamination=configs['isoforest_contamination'], random_state=configs["random_state"])
        iso.fit(X)
        print(f"Using IsolationForest (stateful, contamination={iso.contamination})")
        get_detector = safe_detector(lambda X_data: iso.predict(X_data) == 1) # inliers are labeled 1, outliers -1

    elif method is OutlierDetector.svm:  # One-Class SVM
        # nu ~ contamination, upper bound on fraction of training errors and lower bound of fraction of support vectors.
        clf = OneClassSVM(nu=0.05, kernel='rbf', gamma='scale') # nu in [0, 0.5]
        clf.fit(X)
        print(f"Using One-Class SVM (stateful, nu={clf.nu})")
        get_detector = safe_detector(lambda X_data: clf.predict(X_data) == 1) # inliers 1, outliers -1

    elif method is OutlierDetector.pca_svm:
        # scale data, pca, svm on low-dim representation
        n_components_pca = 2  # hyperparam
        pca_svm_pipeline = make_pipeline(
            StandardScaler(),
            PCA(n_components=configs['pca_n_components'], random_state=configs["random_state"]),
            OneClassSVM(nu=configs['pca_svm_nu'], kernel='rbf', gamma=configs['pca_svm_gamma'])  # rbf is fast on low-dim data
        )

        print(f"Using PCA+SVM detector (stateful, n_components={n_components_pca}, nu={configs['pca_svm_nu']}, gamma={configs['pca_svm_gamma']})")
        pca_svm_pipeline.fit(X)
        get_detector = safe_detector(lambda X_data: pca_svm_pipeline.predict(X_data) == 1) # inliers 1, outliers -1

    elif method is OutlierDetector.pca_isoforest:  # PCA + Isolation Forest
        # IsoForest is not sensitive to scale, so StandardScaler isn't  strictly required for the model, but for PCA.
        pca_isoforest_pipeline = make_pipeline(
            RobustScaler(),
            PCA(n_components=configs['pca_n_components'], random_state=configs["random_state"]),
            IsolationForest(contamination=configs['pca_isoforest_contamination'], random_state=configs["random_state"])
        )
        print(f"Using PCA+IsolationForest detector (stateful, n_components={configs['pca_n_components']}, contamination={configs['pca_isoforest_contamination']})")
        pca_isoforest_pipeline.fit(X)
        get_detector = safe_detector(lambda X_data: pca_isoforest_pipeline.predict(X_data) == 1) # inliers 1, outliers -1

    else:
        raise ValueError(f"Unknown outlier detection method: {method}")

    return get_detector
