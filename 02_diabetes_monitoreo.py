import os
import json
import yaml
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# =====================================================================
# CONFIGURACIÓN Y CONEXIÓN DE MLFLOW
# =====================================================================
MLFLOW_DB_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "Diabetes_Prediction_and_Drift"

try:
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_DB_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger_mlflow = mlflow
    MLFLOW_AVAILABLE = True
    print(f"[MLflow INFO]: Conectado a Tracking URI: {MLFLOW_DB_URI}")
    print(f"[MLflow INFO]: Experimento activo: '{EXPERIMENT_NAME}'")
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
            print("[MLflow Run]: Ejecución finalizada con éxito.")
        def log_param(self, key, value):
            print(f"  -> Param: {key} = {value}")
        def log_metric(self, key, value, step=None):
            print(f"  -> Metric: {key} = {value:.4f}")
        def log_artifact(self, path):
            print(f"  -> Artifact: '{os.path.basename(path)}' registrado.")
            
    logger_mlflow = MockMLflow()

# Configuración visual de gráficos
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
np.random.seed(42)

print("=====================================================================")
print("MLOps: ENTRENAMIENTO, PIPELINE, VERSIONADO CON DVC Y MONITOREO")
print("=====================================================================")

# =====================================================================
# FASE 1: CARGA DE DATOS Y CONFIGURACIÓN DE HIPERPARÁMETROS
# =====================================================================
DATA_PATH = "diabetes.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"No se encontró el dataset en '{DATA_PATH}'. Si estás en otra PC ejecuta: 'dvc pull'")

print(f"\n[1] Cargando dataset de diabetes desde '{DATA_PATH}'...")
df = pd.read_csv(DATA_PATH)
print(f"    Filas: {df.shape[0]}, Columnas: {df.shape[1]}")

# Separación de características y variable objetivo (Outcome: 0 = No diabetes, 1 = Diabetes)
FEATURE_NAMES = [col for col in df.columns if col != 'Outcome']
X = df[FEATURE_NAMES]
y = df['Outcome']

# División estratificada para respetar el balance de clases
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"    Conjunto de Entrenamiento: {X_train.shape[0]} muestras")
print(f"    Conjunto de Evaluación:    {X_test.shape[0]} muestras")

# Definición de hiperparámetros del Random Forest
hyperparameters = {
    "model_type": "RandomForestClassifier",
    "n_estimators": 120,
    "max_depth": 6,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "criterion": "gini",
    "random_state": 42
}

# Guardar hiperparámetros en 'params.yaml' (Estándar de parámetros para DVC)
params_path = "params.yaml"
with open(params_path, "w", encoding="utf-8") as f:
    yaml.dump({"train": hyperparameters}, f, default_flow_style=False)
print(f"    Hiperparámetros guardados exitosamente en '{params_path}'")


# =====================================================================
# FASE 2: CONSTRUCCIÓN DEL PIPELINE (StandardScaler + RandomForest)
# =====================================================================
print("\n[2] Construyendo y entrenando Pipeline de Scikit-Learn...")

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(
        n_estimators=hyperparameters["n_estimators"],
        max_depth=hyperparameters["max_depth"],
        min_samples_split=hyperparameters["min_samples_split"],
        min_samples_leaf=hyperparameters["min_samples_leaf"],
        criterion=hyperparameters["criterion"],
        random_state=hyperparameters["random_state"]
    ))
])

# Entrenamiento del pipeline completo
pipeline.fit(X_train, y_train)
print("    Pipeline entrenado con éxito.")


# =====================================================================
# FASE 3: EVALUACIÓN Y REGISTRO DE MÉTRICAS (DVC Metrics)
# =====================================================================
print("\n[3] Evaluando rendimiento en conjunto de prueba...")
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": round(accuracy_score(y_test, y_pred), 4),
    "precision": round(precision_score(y_test, y_pred), 4),
    "recall": round(recall_score(y_test, y_pred), 4),
    "f1_score": round(f1_score(y_test, y_pred), 4),
    "roc_auc": round(roc_auc_score(y_test, y_pred_proba), 4)
}

print("    Métricas obtenidas:")
for k, v in metrics.items():
    print(f"      - {k.upper():<10}: {v:.4f}")

# Guardar métricas en 'metrics.json' (Estándar de métricas para DVC)
metrics_path = "metrics.json"
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)
print(f"    Métricas exportadas a '{metrics_path}'")

# =====================================================================
# FASE 4: SERIALIZACIÓN Y VERSIONADO DEL MODELO (DVC Model Store)
# =====================================================================
print("\n[4] Guardando modelo entrenado para control de versiones con DVC...")
os.makedirs("models", exist_ok=True)
model_path = os.path.join("models", "diabetes_pipeline.joblib")

joblib.dump(pipeline, model_path)
print(f"    Archivo de modelo exportado: '{model_path}' ({os.path.getsize(model_path)} bytes)")
print("    >> Recomendación DVC: Versiona el modelo pesado con:")
print(f"       dvc add {model_path}")
print(f"       git add {model_path}.dvc params.yaml metrics.json")
print("       git commit -m 'feat: registrar modelo entrenado de diabetes'")

# Registrar en MLflow (Parámetros, Métricas y Artefactos)
with logger_mlflow.start_run(run_name="Entrenamiento_Pipeline_Diabetes"):
    for param_name, param_val in hyperparameters.items():
        logger_mlflow.log_param(param_name, param_val)
    for metric_name, metric_val in metrics.items():
        logger_mlflow.log_metric(metric_name, metric_val)
    logger_mlflow.log_artifact(params_path)
    logger_mlflow.log_artifact(metrics_path)
    logger_mlflow.log_artifact(model_path)


# =====================================================================
# FASE 5: ALGORITMOS DE MONITOREO DE DATA DRIFT (KS-TEST Y PSI)
# =====================================================================
print("\n=====================================================================")
print("FASE 5: MONITOREO DE DATA DRIFT EN CARACTERÍSTICAS CLÍNICAS")
print("=====================================================================")

def calculate_psi(baseline, actual, num_bins=10):
    """
    Calcula el Population Stability Index (PSI) entre dos distribuciones.
    PSI < 0.10: Sin cambio significativo.
    0.10 <= PSI < 0.25: Cambio moderado (monitorear).
    PSI >= 0.25: Drift crítico (reentrenar modelo).
    """
    percentiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(baseline, percentiles)
    bins = np.unique(bins)
    if len(bins) < 2:
        bins = np.linspace(baseline.min(), baseline.max(), num_bins + 1)
        
    baseline_counts, _ = np.histogram(baseline, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)
    
    eps = 1e-4
    psi_base = (baseline_counts + eps) / (len(baseline) + eps * len(baseline_counts))
    psi_act = (actual_counts + eps) / (len(actual) + eps * len(actual_counts))
    
    psi_value = np.sum((psi_act - psi_base) * np.log(psi_act / psi_base))
    return float(psi_value)

def detect_drift_ks(baseline, actual, alpha=0.05):
    """
    Test Kolmogorov-Smirnov de dos muestras.
    Si p_valor < alpha (0.05), se rechaza que provengan de la misma distribución.
    """
    stat, p_value = ks_2samp(baseline, actual)
    drift_detected = bool(p_value < alpha)
    return float(stat), float(p_value), drift_detected

# Seleccionar la característica clínica clave: 'Glucose'
feature_key = 'Glucose'
baseline_glucose = X_train[feature_key].values
mean_base = baseline_glucose.mean()
std_base = baseline_glucose.std()

# Simulación 1: Tráfico en producción estable (pacientes similares a la base histórica)
production_stable = np.random.normal(loc=mean_base, scale=std_base * 0.95, size=250)
production_stable = np.clip(production_stable, 40, 200)

# Simulación 2: Tráfico en producción con Drift (pacientes con hiperglucemia severa o sesgo hospitalario)
production_drifted = np.random.normal(loc=mean_base + 35.0, scale=std_base * 1.15, size=250)
production_drifted = np.clip(production_drifted, 50, 250)

# Evaluar Drift en Tráfico Estable
psi_st = calculate_psi(baseline_glucose, production_stable)
ks_stat_st, ks_p_st, ks_drift_st = detect_drift_ks(baseline_glucose, production_stable)

print(f"\n--- A. Monitoreo Tráfico Estable ({feature_key}) ---")
print(f"    PSI:     {psi_st:.4f}  (Umbral de alerta >= 0.25)")
print(f"    KS-Test: Estadístico={ks_stat_st:.4f}, p-valor={ks_p_st:.4f} (Drift detectado: {ks_drift_st})")
print("    ESTADO:  Tráfico normal y estable. El modelo opera con confianza.")

# Evaluar Drift en Tráfico Desviado
psi_dr = calculate_psi(baseline_glucose, production_drifted)
ks_stat_dr, ks_p_dr, ks_drift_dr = detect_drift_ks(baseline_glucose, production_drifted)

print(f"\n--- B. Monitoreo Tráfico Desviado / Drift ({feature_key}) ---")
print(f"    PSI:     {psi_dr:.4f}  (ALERTA CRÍTICA: Distribución poblacional alterada)")
print(f"    KS-Test: Estadístico={ks_stat_dr:.4f}, p-valor={ks_p_dr:.8f} (Drift detectado: {ks_drift_dr})")
print("    ¡ALERTA!: Degradación inminente del modelo detectada.")
print("    --> Solución MLOps: Desencadenar pipeline de reentrenamiento continuo (CT) y versionar nuevo modelo con DVC.")


# =====================================================================
# FASE 6: REPORTE VISUAL DE MONITOREO
# =====================================================================
print("\n[6] Generando gráfico de reporte de Data Drift...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Subgráfico 1: Tráfico Estable vs Baseline
ax1.hist(baseline_glucose, bins=20, alpha=0.45, color="steelblue", label="Baseline (Entrenamiento)", density=True)
ax1.hist(production_stable, bins=20, alpha=0.8, color="forestgreen", label="Producción Estable", density=True, histtype='step', linewidth=2.5)
ax1.set_title(f"A. Tráfico Estable - {feature_key}\nPSI: {psi_st:.4f} | KS p-val: {ks_p_st:.3f}", fontsize=12, fontweight='bold')
ax1.set_xlabel("Nivel de Glucosa (mg/dL)")
ax1.set_ylabel("Densidad")
ax1.legend()

# Subgráfico 2: Tráfico Desviado vs Baseline
ax2.hist(baseline_glucose, bins=20, alpha=0.45, color="steelblue", label="Baseline (Entrenamiento)", density=True)
ax2.hist(production_drifted, bins=20, alpha=0.8, color="crimson", label="Producción con Drift", density=True, histtype='step', linewidth=2.5)
ax2.set_title(f"B. Tráfico Desviado - {feature_key}\nPSI: {psi_dr:.4f} (ALERTA) | KS p-val: {ks_p_dr:.2e}", fontsize=12, fontweight='bold', color="darkred")
ax2.set_xlabel("Nivel de Glucosa (mg/dL)")
ax2.set_ylabel("Densidad")
ax2.legend()

plt.tight_layout()
report_path = "diabetes_drift_report.png"
plt.savefig(report_path, dpi=150)
plt.close()

print(f"    Reporte guardado exitosamente como: '{report_path}'")

# Registrar artefacto en MLflow
with logger_mlflow.start_run(run_name="Monitoreo_Drift_Diabetes"):
    logger_mlflow.log_metric("glucose_psi_stable", psi_st)
    logger_mlflow.log_metric("glucose_psi_drifted", psi_dr)
    logger_mlflow.log_artifact(report_path)

print("\n=====================================================================")
print("PROCESO COMPLETADO CON ÉXITO")
print("=====================================================================")
