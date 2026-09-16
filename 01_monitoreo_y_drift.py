import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

# =====================================================================
# SIMULADOR DE MLFLOW (Para ejecutar sin dependencias complejas)
# =====================================================================
try:
    import mlflow
    logger_mlflow = mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    class MockMLflow:
        def __init__(self):
            print("[MLflow INFO]: Cargado en modo 'Mock/Simulación' (MLflow no instalado localmente).")
        def start_run(self, run_name=None):
            print(f"\n[MLflow Run]: Iniciando experimento - '{run_name}'")
            return self
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            print("[MLflow Run]: Ejecución finalizada con éxito y guardada en el Registro.")
        def log_param(self, key, value):
            print(f"  -> Logged Param: {key} = {value}")
        def log_metric(self, key, value, step=None):
            print(f"  -> Logged Metric: {key} = {value:.4f}")
        def log_artifact(self, path):
            print(f"  -> Logged Artifact: '{os.path.basename(path)}' registrado en Model Registry.")
            
    logger_mlflow = MockMLflow()

# Configuración del gráfico
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
np.random.seed(42)

print("=====================================================================")
print("FASE 1: SIMULACIÓN DE MLOPS CON SEGUIMIENTO DE EXPERIMENTOS")
print("=====================================================================")

# Registrar parámetros en MLflow
with logger_mlflow.start_run(run_name="Entrenamiento_Base_Iris") as run:
    logger_mlflow.log_param("model_type", "RandomForestClassifier")
    logger_mlflow.log_param("n_estimators", 100)
    logger_mlflow.log_param("random_state", 42)
    
    # Simular métricas de validación loggeadas
    logger_mlflow.log_metric("train_accuracy", 0.985)
    logger_mlflow.log_metric("val_accuracy", 0.950)
    logger_mlflow.log_metric("val_f1_score", 0.948)


# =====================================================================
# FASE 2: ALGORITMOS DE DETECCIÓN DE DATA DRIFT
# =====================================================================
print("\n=====================================================================")
print("FASE 2: ALGORITMOS MATEMÁTICOS DE DATA DRIFT (KS-TEST Y PSI)")
print("=====================================================================")

def calculate_psi(baseline, actual, num_bins=10):
    """
    Calcula el Population Stability Index (PSI) entre dos distribuciones.
    """
    # Establecer los límites de los cubos (bins) basados en el baseline
    percentiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(baseline, percentiles)
    # Evitar bordes duplicados ajustando con pequeña varianza si es necesario
    bins = np.unique(bins)
    if len(bins) < 2:
        bins = np.linspace(baseline.min(), baseline.max(), num_bins + 1)
        
    # Calcular frecuencias en los cubos
    baseline_counts, _ = np.histogram(baseline, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)
    
    # Convertir a proporciones (probabilidades) añadiendo epsilon para evitar división por cero
    eps = 1e-4
    psi_base = (baseline_counts + eps) / (len(baseline) + eps * len(baseline_counts))
    psi_act = (actual_counts + eps) / (len(actual) + eps * len(actual_counts))
    
    # Calcular ecuación de PSI
    psi_value = np.sum((psi_act - psi_base) * np.log(psi_act / psi_base))
    return psi_value

def detect_drift_ks(baseline, actual, alpha=0.05):
    """
    Realiza el test de Kolmogorov-Smirnov de dos muestras para verificar drift.
    """
    stat, p_value = ks_2samp(baseline, actual)
    drift_detected = p_value < alpha
    return stat, p_value, drift_detected


# =====================================================================
# FASE 3: SIMULACIÓN DE TRÁFICO PRODUCTIVO Y DRIFT
# =====================================================================
print("\nGenerando datos históricos de entrenamiento (Baseline)...")
# Supongamos una característica clave: 'petal_length' de flores iris versicolor/virginica
baseline_petal_length = np.random.normal(loc=4.5, scale=0.5, size=500)

print("\n--- SIMULACIÓN A: TRÁFICO ESTABLE (Sin Drift) ---")
# Datos entrantes sin cambios significativos en el comportamiento
production_stable = np.random.normal(loc=4.48, scale=0.48, size=200)

# Evaluar Drift
psi_stable = calculate_psi(baseline_petal_length, production_stable)
ks_stat_st, ks_p_st, ks_drift_st = detect_drift_ks(baseline_petal_length, production_stable)

print(f"PSI (Estable): {psi_stable:.4f}")
print(f"KS-Test (Estable): Estadístico = {ks_stat_st:.4f}, p-valor = {ks_p_st:.4f} (Drift: {ks_drift_st})")
if psi_stable >= 0.25 or ks_drift_st:
    print("ALERTA: Se detecta Data Drift en Tráfico Estable. Reentrenar modelo.")
else:
    print("ESTADO: Tráfico estable. No se requiere acción.")

print("\n--- SIMULACIÓN B: TRÁFICO DESVIADO (Con Data Drift) ---")
# Cambia la población de entrada: la media sube a 5.2 (ej. cambio climático o nuevas especies)
production_drifted = np.random.normal(loc=5.2, scale=0.6, size=200)

# Evaluar Drift
psi_drifted = calculate_psi(baseline_petal_length, production_drifted)
ks_stat_dr, ks_p_dr, ks_drift_dr = detect_drift_ks(baseline_petal_length, production_drifted)

print(f"PSI (Desviado): {psi_drifted:.4f}")
print(f"KS-Test (Desviado): Estadístico = {ks_stat_dr:.4f}, p-valor = {ks_p_dr:.4f} (Drift: {ks_drift_dr})")
if psi_drifted >= 0.25 or ks_drift_dr:
    print("¡ALERTA CRÍTICA!: Data Drift detectado de forma significativa.")
    print("--> Acción sugerida: Iniciar automáticamente pipeline de Reentrenamiento (CT).")
else:
    print("ESTADO: Tráfico estable. No se requiere acción.")


# =====================================================================
# FASE 4: REPORTABILIDAD VISUAL (DISTRIBUCIÓN Y DRIFT)
# =====================================================================
# Generar gráfico comparativo de distribuciones
fig, ax = plt.subplots(figsize=(10, 6))

# Dibujar Baseline
ax.hist(baseline_petal_length, bins=25, alpha=0.4, color="blue", label="Baseline (Entrenamiento)", density=True)
# Dibujar Estable
ax.hist(production_stable, bins=25, alpha=0.4, color="green", label="Producción Estable", density=True, histtype='step', linewidth=2)
# Dibujar Desviado
ax.hist(production_drifted, bins=25, alpha=0.4, color="red", label="Producción Desviada (Drift)", density=True, histtype='step', linewidth=2)

ax.set_title("Monitoreo de Data Drift: Distribución de Petal Length", fontsize=14, fontweight='bold')
ax.set_xlabel("Valor de Característica (cm)")
ax.set_ylabel("Densidad de Probabilidad")
ax.legend(loc="upper right")

# Añadir resúmenes estadísticos al gráfico
info_text = (
    f"Tráfico Estable:\n  PSI = {psi_stable:.4f}\n  KS p-val = {ks_p_st:.4f}\n\n"
    f"Tráfico Desviado:\n  PSI = {psi_drifted:.4f} (CRÍTICO)\n  KS p-val = {ks_p_dr:.8f}"
)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

plt.tight_layout()
report_image_path = "drift_report.png"
plt.savefig(report_image_path)
plt.close()

print("\n=====================================================================")
print("REGISTRO DE ARTEFACTOS Y FIN DEL PROCESO")
print("=====================================================================")
print(f"Gráfico de reporte de Data Drift exportado exitosamente como: '{report_image_path}'")

# Registrar el reporte visual en MLflow
with logger_mlflow.start_run(run_name="Evaluacion_Monitoreo_Drift") as run:
    logger_mlflow.log_metric("stable_psi", psi_stable)
    logger_mlflow.log_metric("drifted_psi", psi_drifted)
    logger_mlflow.log_artifact(report_image_path)
