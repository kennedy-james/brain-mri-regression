"""
Run inference with trained models.
"""
from typing import Dict, Any

import numpy as np


def predict_with_pipeline(x: np.ndarray, pipeline: Dict[str, Any]) -> np.ndarray:
    """Run full pipeline: imputation -> selection -> model.predict."""
    imputer = pipeline["imputer"]
    selection = pipeline["selection"]
    model = pipeline["model"]

    x_imp = imputer.transform(x)
    x_sel = selection.transform(x_imp)
    preds = model.predict(x_sel)
    return np.asarray(preds).ravel()