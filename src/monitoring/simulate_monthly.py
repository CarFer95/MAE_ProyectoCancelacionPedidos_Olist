from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.pipeline import Pipeline


def simular_nuevos_datos_mensuales(
    df_base: pd.DataFrame,
    pipeline: Pipeline,
    target_col: str,
    date_col: str = "order_purchase_timestamp",
    features_drop: Iterable[str] | None = None,
    drift_threshold: float = 0.02,
) -> pd.DataFrame:
    """
    Simulación mensual + evaluación por mes + detección de drift.
    Usa la columna date_col para agrupar año-mes (purchase_ym).
    """
    if features_drop is None:
        features_drop = [
            "order_id",
            "order_status",
            "order_purchase_timestamp",
            "is_canceled_strict",
            "order_canceled_extended",
        ]

    df = df_base.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    if "purchase_ym" not in df.columns:
        df["purchase_ym"] = (
            df[date_col].dt.to_period("M").astype(str).str.replace("-", "")
        )

    resultados = []

    for mes, df_mes in df.groupby("purchase_ym"):
        df_mes = df_mes.copy()

        y = df_mes[target_col].astype(int).values
        X = df_mes.drop(columns=list(features_drop))

        if len(df_mes) == 0:
            continue

        y_proba = pipeline.predict_proba(X)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        cancel_real = y.mean()
        cancel_pred = (y_pred == 1).mean()

        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        rec = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        gap = abs(cancel_real - cancel_pred)

        resultados.append(
            {
                "purchase_ym": mes,
                "n_pedidos_mes": len(df_mes),
                "cancel_rate_real": cancel_real,
                "cancel_rate_predicha": cancel_pred,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "gap_cancel_rate": gap,
                "alerta_drift": "ALERTA" if gap > drift_threshold else "OK",
            }
        )

    monitor_mensual = pd.DataFrame(resultados).sort_values("purchase_ym")
    return monitor_mensual
