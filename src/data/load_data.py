from pathlib import Path
import pandas as pd


def load_orders_extended(
    path: str | Path = "data/processed/orders_extended_for_eda.csv",
) -> pd.DataFrame:
    """
    Carga el dataset extendido (orders_extended_for_eda.csv) con el que entrenas el modelo.
    Asegura que la columna de fecha esté en datetime.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo procesado en {path}. "
            "Asegúrate de generarlo primero con build_dataset.py."
        )

    df = pd.read_csv(path, parse_dates=["order_purchase_timestamp"])
    return df
