<<<<<<< HEAD
# MAE_ProyectoCancelacionPedidos_olist
=======
# Predicción de Cancelaciones – Olist

Proyecto orientado a predecir cancelaciones de pedidos (`order_canceled_extended`)
y simular la incorporación mensual de nuevos datos, usando el dataset público de Olist.

## Estructura

- `data/raw/`: CSV originales de Olist (no se versionan).
- `data/processed/`: datasets procesados (`orders_extended_for_eda.csv`, `monitor_mensual.csv`).
- `src/`: módulos de datos, features, modelos y monitoreo.
- `models/`: artefactos entrenados (`cancel_model.joblib`).
- `notebooks/`: notebooks de exploración (EDA, pruebas).
- `docs/`: documentación del flujo de datos y versionamiento.

## Requisitos

```bash
pip install -r requirements.txt
```

## Pasos para reproducir

1. Descargar los CSV de Olist y copiarlos a `data/raw/`.
2. Construir el dataset extendido:

   ```bash
   python -m src.data.build_dataset
   ```

3. Ejecutar el pipeline completo (entrenar + simular mensual):

   ```bash
   python run_from_extended_dataset.py
   ```

4. Revisar:

   - Modelo entrenado: `models/cancel_model.joblib`
   - Monitor mensual: `data/processed/monitor_mensual.csv`
>>>>>>> 2059b2b (Proyecto modularizado inicial)
