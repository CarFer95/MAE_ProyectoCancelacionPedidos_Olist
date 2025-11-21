from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)

from src.features.select_features import prepare_xy
from src.models.pipeline import create_model_pipeline


TRAIN_MONTHS_DEFAULT: List[str] = [
    "201610",
    "201611",
    "201612",
    *[f"2017{str(m).zfill(2)}" for m in range(1, 13)],
    "201801",
    "201802",
    "201803",
    "201804",
]

BACKTEST_MONTHS_DEFAULT: List[str] = ["201805", "201806", "201807"]
FINAL_TEST_MONTHS_DEFAULT: List[str] = ["201808"]


def split_by_month(
    df: pd.DataFrame,
    train_months: List[str],
    backtest_months: List[str],
    final_test_months: List[str] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Separa el DataFrame en train, backtest y final_test usando purchase_ym (YYYYMM)."""
    if final_test_months is None:
        final_test_months = []

    df_train = df[df["purchase_ym"].isin(train_months)].copy()
    df_backtest = df[df["purchase_ym"].isin(backtest_months)].copy()
    df_final = df[df["purchase_ym"].isin(final_test_months)].copy()

    return df_train, df_backtest, df_final


def train_and_save_model(
    df: pd.DataFrame,
    model_path: str | Path = "models/cancel_model.joblib",
    target_col: str = "order_canceled_extended",
    train_months: List[str] | None = None,
    backtest_months: List[str] | None = None,
    final_test_months: List[str] | None = None,
) -> Dict[str, float]:
    """
    Entrena el pipeline de Regresión Logística con split temporal y guarda el modelo.
    Devuelve métricas en el backtest (recall, F1, AUC, Gini).
    """
    if train_months is None:
        train_months = TRAIN_MONTHS_DEFAULT
    if backtest_months is None:
        backtest_months = BACKTEST_MONTHS_DEFAULT
    if final_test_months is None:
        final_test_months = FINAL_TEST_MONTHS_DEFAULT

    df_train, df_backtest, df_final = split_by_month(
        df, train_months, backtest_months, final_test_months
    )

    print(f"Train: {df_train.shape}, Backtest: {df_backtest.shape}, Final: {df_final.shape}")

    # X / y
    X_train, y_train = prepare_xy(df_train, target_col=target_col)
    X_back, y_back = prepare_xy(df_backtest, target_col=target_col)

    # Separar columnas numéricas y categóricas
    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = [c for c in X_train.columns if c not in numeric_features]

    clf = create_model_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    print("Entrenando modelo...")
    clf.fit(X_train, y_train)
    print("Entrenamiento completado.")

    # Evaluación en backtest
    y_pred = clf.predict(X_back)
    y_proba = clf.predict_proba(X_back)[:, 1]

    rec = recall_score(y_back, y_pred)
    f1 = f1_score(y_back, y_pred)
    auc = roc_auc_score(y_back, y_proba)
    gini = 2 * auc - 1

    print("\nMÉTRICAS EN BACKTEST (target extendido)")
    print("----------------------------------------")
    print(f"Recall:   {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC:  {auc:.4f}")
    print(f"Gini:     {gini:.4f}")
    print("\nMatriz de confusión:")
    print(confusion_matrix(y_back, y_pred))
    print("\nReporte de clasificación:")
    print(classification_report(y_back, y_pred, digits=4))

    # Guardar modelo
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"\nModelo guardado en: {model_path}")

    return {
        "recall_backtest": rec,
        "f1_backtest": f1,
        "auc_backtest": auc,
        "gini_backtest": gini,
    }
