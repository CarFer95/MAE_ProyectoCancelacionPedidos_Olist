# 01 – Flujo de Datos

1. **Fuente**: CSV de Olist (`data/raw/`).
2. **Ingesta y construcción**: `src/data/build_dataset.py`
   - Unión de tablas.
   - Creación de target extendido (`order_canceled_extended`).
3. **Dataset maestro**: `data/processed/orders_extended_for_eda.csv`
4. **Feature Engineering**: `src/features/make_features.py`
5. **Split temporal y entrenamiento**: `src/models/train_model.py`
6. **Monitoreo mensual**: `src/monitoring/simulate_monthly.py`
7. **Salida de monitoreo**: `data/processed/monitor_mensual.csv`
