from pathlib import Path

import joblib
import matplotlib.pyplot as plt

from src.data.load_data import load_orders_extended
from src.features.make_features import create_features
from src.models.train_model import train_and_save_model
from src.monitoring.simulate_monthly import simular_nuevos_datos_mensuales


def main():
    # 1) Cargar dataset extendido
    df = load_orders_extended("data/processed/orders_extended_for_eda.csv")

    # 2) Features (purchase_ym, etc.)
    df = create_features(df)

    # 3) Entrenar y guardar modelo
    metrics = train_and_save_model(
        df=df,
        model_path="models/cancel_model.joblib",
        target_col="order_canceled_extended",
    )
    print("\nMétricas backtest:", metrics)

    # 4) Cargar modelo y simular incorporación mensual
    model_path = Path("models/cancel_model.joblib")
    clf = joblib.load(model_path)

    features_drop = [
        "order_id",
        "order_status",
        "order_purchase_timestamp",
        "is_canceled_strict",
        "order_canceled_extended",
    ]

    monitor_mensual = simular_nuevos_datos_mensuales(
        df_base=df,
        pipeline=clf,
        target_col="order_canceled_extended",
        date_col="order_purchase_timestamp",
        features_drop=features_drop,
        drift_threshold=0.02,
    )

    # 5) Guardar monitor mensual
    out_path = Path("data/processed/monitor_mensual.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    monitor_mensual.to_csv(out_path, index=False)
    print(f"\nMonitor mensual guardado en: {out_path}")

    # 6) Gráfico simple
    plt.figure(figsize=(10, 5))
    plt.plot(
        monitor_mensual["purchase_ym"],
        monitor_mensual["cancel_rate_real"],
        marker="o",
        label="Real",
    )
    plt.plot(
        monitor_mensual["purchase_ym"],
        monitor_mensual["cancel_rate_predicha"],
        marker="s",
        linestyle="--",
        label="Predicha",
    )
    plt.xticks(rotation=45)
    plt.ylabel("Tasa de cancelación")
    plt.title(
        "Simulación de incorporación mensual de nuevos datos\n"
        "Tasa de cancelación real vs predicha"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
