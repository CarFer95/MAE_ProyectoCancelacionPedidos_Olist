from typing import Iterable, Tuple
import numpy as np
import pandas as pd


def prepare_xy(
    df: pd.DataFrame,
    target_col: str = "order_canceled_extended",
    features_drop: Iterable[str] | None = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Separa X (features) e y (target).
    Por defecto, descarta: order_id, order_status, order_purchase_timestamp,
    is_canceled_strict, order_canceled_extended.
    """
    if features_drop is None:
        features_drop = [
            "order_id",
            "order_status",
            "order_purchase_timestamp",
            "is_canceled_strict",
            "order_canceled_extended",
        ]

    df = df.copy()
    y = df[target_col].astype(int).values
    X = df.drop(columns=list(features_drop))

    return X, y
