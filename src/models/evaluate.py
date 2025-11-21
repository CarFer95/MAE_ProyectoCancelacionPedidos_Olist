from typing import Dict

import numpy as np
from sklearn.metrics import (
    f1_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, float]:
    """Calcula métricas básicas de clasificación binaria."""
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    gini = 2 * auc - 1

    return {
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "gini": gini,
    }
