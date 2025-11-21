from pathlib import Path
import numpy as np
import pandas as pd


def moda_safe(series: pd.Series):
    """Devuelve la moda o NaN si no existe."""
    if series.isna().all():
        return np.nan
    m = series.mode()
    if len(m) == 0:
        return np.nan
    return m.iloc[0]


def build_orders_extended(
    raw_dir: str | Path = "data/raw",
    output_path: str | Path = "data/processed/orders_extended_for_eda.csv",
) -> pd.DataFrame:
    """
    Construye el dataset extendido a nivel pedido:
    - Une ORDERS + ITEMS agregados + pagos + customers + sellers + categorías.
    - Crea target estricto y extendido (order_canceled_extended).
    - Deja todo listo para EDA y modelado.
    """
    raw_dir = Path(raw_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Cargar tablas base ---
    orders = pd.read_csv(
        raw_dir / "olist_orders_dataset.csv",
        parse_dates=[
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )

    order_items = pd.read_csv(raw_dir / "olist_order_items_dataset.csv")
    order_payments = pd.read_csv(raw_dir / "olist_order_payments_dataset.csv")
    customers = pd.read_csv(raw_dir / "olist_customers_dataset.csv")
    sellers = pd.read_csv(raw_dir / "olist_sellers_dataset.csv")
    products = pd.read_csv(raw_dir / "olist_products_dataset.csv")
    cat_trad = pd.read_csv(raw_dir / "product_category_name_translation.csv")

    # --- Target estricto y extendido ---
    orders["is_canceled_strict"] = (orders["order_status"] == "canceled").astype(int)

    cond_canceled = orders["order_status"] == "canceled"
    cond_unavailable = orders["order_status"] == "unavailable"

    # created/processing “colgados”
    cond_created_processing = orders["order_status"].isin(["created", "processing"])
    cond_sin_entrega_30d = (
        orders["order_delivered_customer_date"].isna()
        & (
            (
                orders["order_purchase_timestamp"]
                + pd.Timedelta(days=30)
            )
            < orders["order_estimated_delivery_date"].fillna(
                orders["order_purchase_timestamp"] + pd.Timedelta(days=30)
            )
        )
    )
    cond_colgados = cond_created_processing & cond_sin_entrega_30d

    cond_invoiced = orders["order_status"] == "invoiced"
    cond_shipped = orders["order_status"] == "shipped"
    cond_aprobado_sin_envio = orders["order_approved_at"].notna() & (
        orders["order_delivered_carrier_date"].isna()
    )

    orders["order_canceled_extended"] = np.where(
        cond_canceled
        | cond_unavailable
        | cond_colgados
        | (cond_invoiced & cond_aprobado_sin_envio)
        | (cond_shipped & cond_aprobado_sin_envio),
        1,
        0,
    )

    # --- Productos + traducción de categorías ---
    products_trad = products.merge(
        cat_trad,
        on="product_category_name",
        how="left",
    ).rename(
        columns={
            "product_category_name": "product_category_name_pt",
            "product_category_name_english": "product_category_name_en",
        }
    )

    # Items con info de producto
    order_items_full = order_items.merge(
        products_trad, on="product_id", how="left"
    )

    # Agregados de items por pedido
    items_agg = (
        order_items_full.groupby("order_id")
        .agg(
            price_sum=("price", "sum"),
            freight_sum=("freight_value", "sum"),
            items_qty=("order_item_id", "count"),
            seller_count=("seller_id", "nunique"),
        )
        .reset_index()
    )

    items_cat = (
        order_items_full.groupby("order_id")
        .agg(
            main_category=("product_category_name_pt", moda_safe),
            main_category_en=("product_category_name_en", moda_safe),
        )
        .reset_index()
    )

    # Pagos agregados
    pay_agg = (
        order_payments.groupby("order_id")
        .agg(
            payment_value_sum=("payment_value", "sum"),
            max_installments=("payment_installments", "max"),
            payments_cnt=("payment_sequential", "count"),
            main_payment_type=("payment_type", moda_safe),
        )
        .reset_index()
    )

    # ORDERS + ITEMS agregados + CATEGORÍAS
    orders_items = (
        orders.merge(items_agg, on="order_id", how="left")
        .merge(items_cat, on="order_id", how="left")
    )

    # + Pagos
    orders_items_pay = orders_items.merge(pay_agg, on="order_id", how="left")

    # + Clientes
    orders_full = orders_items_pay.merge(
        customers[["customer_id", "customer_city", "customer_state"]],
        on="customer_id",
        how="left",
    )

    # + Ubicación principal del vendedor
    seller_loc = (
        order_items_full.merge(
            sellers[["seller_id", "seller_city", "seller_state"]],
            on="seller_id",
            how="left",
        )
        .groupby("order_id")
        .agg(
            main_seller_city=("seller_city", moda_safe),
            main_seller_state=("seller_state", moda_safe),
        )
        .reset_index()
    )

    orders_full = orders_full.merge(seller_loc, on="order_id", how="left")

    # Derivadas clave
    orders_full["order_total_value"] = (
        orders_full["price_sum"].fillna(0) + orders_full["freight_sum"].fillna(0)
    )
    orders_full["purchase_ym"] = (
        orders_full["order_purchase_timestamp"].dt.to_period("M").astype(str)
    )

    # Dataset final
    df_extended = orders_full.copy()

    # Guardar
    df_extended.to_csv(output_path, index=False)
    print(f"Dataset extendido guardado en: {output_path}")
    return df_extended


if __name__ == "__main__":
    build_orders_extended()
