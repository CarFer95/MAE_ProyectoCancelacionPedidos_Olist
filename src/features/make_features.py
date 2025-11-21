import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica transformaciones de feature engineering simples y asegura
    la columna purchase_ym (YYYYMM) para el split temporal.
    Asume que df ya tiene order_canceled_extended y otras columnas creadas.
    """
    df = df.copy()

    # Asegurar tipo datetime
    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"], errors="coerce"
    )

    # Año-mes en formato YYYYMM para el split
    df["purchase_ym"] = (
        df["order_purchase_timestamp"].dt.to_period("M")
        .astype(str)
        .str.replace("-", "")
    )

    return df
