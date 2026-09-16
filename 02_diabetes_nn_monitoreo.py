import os
import json
import yaml
import joblib
import webbrowser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Desactivar logs innecesarios de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback

import wandb

# =====================================================================
# CONFIGURACIÓN DE WEIGHTS & BIASES (W&B)
# =====================================================================
WANDB_PROJECT = "diabetes-nn-mlops"

# Verificar si hay credenciales de W&B disponibles
def get_wandb_mode():
    api_key = os.getenv("WANDB_API_KEY")
    if api_key:
        return "online"
    try:
        if wandb.api.api_key:
            return "online"
    except Exception:
        pass
    return "offline"

WANDB_MODE = get_wandb_mode()
print("=====================================================================")
print("MLOps: REDES NEURONALES CON TENSORFLOW, TRACKING CON W&B Y DVC")
print("=====================================================================")
print(f"[W&B INFO]: Modo de seguimiento: '{WANDB_MODE.upper()}'")
if WANDB_MODE == "offline":
    print("  -> (Nota: Para visualizar el dashboard en la nube, ejecuta 'wandb login' con tu clave de https://wandb.ai/authorize)")
    print("  -> Actualmente las métricas y gráficos se guardarán de forma local y segura.\n")

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
tf.random.set_seed(42)
np.random.seed(42)


# =====================================================================
# FASE 1: CARGA DE DATOS Y PREPROCESAMIENTO
# =====================================================================
DATA_PATH = "diabetes.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"No se encontró '{DATA_PATH}'. Si estás en otra máquina ejecuta: 'dvc pull'")

print(f"\n[1] Cargando dataset '{DATA_PATH}'...")
df = pd.read_csv(DATA_PATH)

FEATURE_NAMES = [col for col in df.columns if col != 'Outcome']
X = df[FEATURE_NAMES]
y = df['Outcome']

# Partición Estratificada 80% Train, 20% Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Estandarización con StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Guardar el scaler para inferencia productiva
os.makedirs("models", exist_ok=True)
scaler_path = os.path.join("models", "scaler_nn.joblib")
joblib.dump(scaler, scaler_path)
print(f"    Scaler guardado en: '{scaler_path}'")
print(f"    Muestras de entrenamiento: {X_train_scaled.shape[0]}, Prueba: {X_test_scaled.shape[0]}")


# =====================================================================
# CALLBACK PERSONALIZADO DE MÉTRICAS PARA W&B
# =====================================================================
class WandbEpochLogger(Callback):
    """Registra las métricas por época directamente en Weights & Biases."""
    def on_epoch_end(self, epoch, logs=None):
        if logs and wandb.run is not None:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": logs.get("loss"),
                "train_accuracy": logs.get("accuracy"),
                "val_loss": logs.get("val_loss"),
                "val_accuracy": logs.get("val_accuracy")
            })


# =====================================================================
# FASE 2: DEFINICIÓN, ENTRENAMIENTO Y TRACKING DE 2 MODELOS DE RED NEURONAL
# =====================================================================
input_dim = X_train_scaled.shape[1]

# ---------------------------------------------------------------------
# MODELO 1: Red Neuronal Simple (Baseline Shallow NN)
# ---------------------------------------------------------------------
config_m1 = {
    "model_name": "Modelo_1_Shallow_NN",
    "architecture": "Dense(32, relu) -> Dropout(0.2) -> Dense(16, relu) -> Dense(1, sigmoid)",
    "learning_rate": 0.005,
    "batch_size": 32,
    "epochs": 35,
    "optimizer": "Adam",
    "dropout": 0.2
}

print("\n" + "="*65)
print("[2.1] ENTRENANDO MODELO 1: Shallow Neural Network")
print("="*65)

model_1 = Sequential([
    Dense(32, activation='relu', input_shape=(input_dim,), name="dense_1"),
    Dropout(config_m1["dropout"], name="dropout_1"),
    Dense(16, activation='relu', name="dense_2"),
    Dense(1, activation='sigmoid', name="output")
], name="Shallow_NN")

model_1.compile(
    optimizer=Adam(learning_rate=config_m1["learning_rate"]),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Inicializar corrida en Weights & Biases para Modelo 1
run_1 = wandb.init(
    project=WANDB_PROJECT,
    name=config_m1["model_name"],
    config=config_m1,
    mode=WANDB_MODE,
    reinit=True
)

history_1 = model_1.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=config_m1["epochs"],
    batch_size=config_m1["batch_size"],
    callbacks=[WandbEpochLogger()],
    verbose=0
)

# Evaluación en Test de Modelo 1
y_pred_prob_1 = model_1.predict(X_test_scaled, verbose=0).ravel()
y_pred_1 = (y_pred_prob_1 >= 0.5).astype(int)

metrics_m1 = {
    "test_accuracy": round(float(accuracy_score(y_test, y_pred_1)), 4),
    "test_precision": round(float(precision_score(y_test, y_pred_1)), 4),
    "test_recall": round(float(recall_score(y_test, y_pred_1)), 4),
    "test_f1_score": round(float(f1_score(y_test, y_pred_1)), 4),
    "test_roc_auc": round(float(roc_auc_score(y_test, y_pred_prob_1)), 4)
}

wandb.log(metrics_m1)
run_1_url = run_1.get_url() if WANDB_MODE == "online" else None
run_1.finish()

print(f"    Métricas Modelo 1 (Shallow):")
for k, v in metrics_m1.items():
    print(f"      - {k:<15}: {v:.4f}")


# ---------------------------------------------------------------------
# MODELO 2: Red Neuronal Profunda Regularizada (Deep NN con BatchNorm)
# ---------------------------------------------------------------------
config_m2 = {
    "model_name": "Modelo_2_Deep_BatchNormalized_NN",
    "architecture": "Dense(64) -> BN -> Dropout(0.3) -> Dense(32) -> BN -> Dropout(0.2) -> Dense(16) -> Dense(1)",
    "learning_rate": 0.001,
    "batch_size": 16,
    "epochs": 50,
    "optimizer": "Adam",
    "dropout_1": 0.3,
    "dropout_2": 0.2
}

print("\n" + "="*65)
print("[2.2] ENTRENANDO MODELO 2: Deep & BatchNormalized Neural Network")
print("="*65)

model_2 = Sequential([
    Dense(64, activation='relu', input_shape=(input_dim,), name="dense_1"),
    BatchNormalization(name="bn_1"),
    Dropout(config_m2["dropout_1"], name="dropout_1"),
    Dense(32, activation='relu', name="dense_2"),
    BatchNormalization(name="bn_2"),
    Dropout(config_m2["dropout_2"], name="dropout_2"),
    Dense(16, activation='relu', name="dense_3"),
    Dense(1, activation='sigmoid', name="output")
], name="Deep_NN")

model_2.compile(
    optimizer=Adam(learning_rate=config_m2["learning_rate"]),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Inicializar corrida en Weights & Biases para Modelo 2
run_2 = wandb.init(
    project=WANDB_PROJECT,
    name=config_m2["model_name"],
    config=config_m2,
    mode=WANDB_MODE,
    reinit=True
)

history_2 = model_2.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=config_m2["epochs"],
    batch_size=config_m2["batch_size"],
    callbacks=[WandbEpochLogger()],
    verbose=0
)

# Evaluación en Test de Modelo 2
y_pred_prob_2 = model_2.predict(X_test_scaled, verbose=0).ravel()
y_pred_2 = (y_pred_prob_2 >= 0.5).astype(int)

metrics_m2 = {
    "test_accuracy": round(float(accuracy_score(y_test, y_pred_2)), 4),
    "test_precision": round(float(precision_score(y_test, y_pred_2)), 4),
    "test_recall": round(float(recall_score(y_test, y_pred_2)), 4),
    "test_f1_score": round(float(f1_score(y_test, y_pred_2)), 4),
    "test_roc_auc": round(float(roc_auc_score(y_test, y_pred_prob_2)), 4)
}

wandb.log(metrics_m2)
run_2_url = run_2.get_url() if WANDB_MODE == "online" else None
run_2.finish()

print(f"    Métricas Modelo 2 (Deep):")
for k, v in metrics_m2.items():
    print(f"      - {k:<15}: {v:.4f}")


# =====================================================================
# FASE 3: COMPARACIÓN Y SELECCIÓN DEL MEJOR MODELO
# =====================================================================
print("\n" + "="*65)
print("[3] COMPARATIVA DE RENDIMIENTO ENTRE MODELOS (W&B Benchmark)")
print("="*65)
comparison_df = pd.DataFrame([metrics_m1, metrics_m2], index=["Shallow_NN", "Deep_BN_NN"])
print(comparison_df.to_string())

# Guardar ambos modelos
m1_path = os.path.join("models", "diabetes_nn_shallow.keras")
m2_path = os.path.join("models", "diabetes_nn_deep.keras")
model_1.save(m1_path)
model_2.save(m2_path)
print(f"\n    Modelos Keras guardados exitosamente:")
print(f"      - '{m1_path}' ({os.path.getsize(m1_path)} bytes)")
print(f"      - '{m2_path}' ({os.path.getsize(m2_path)} bytes)")

# Elegir mejor modelo según ROC-AUC
if metrics_m2["test_roc_auc"] >= metrics_m1["test_roc_auc"]:
    best_model_name = "Deep_BN_NN"
    best_metrics = metrics_m2
    best_config = config_m2
    best_model_path = m2_path
else:
    best_model_name = "Shallow_NN"
    best_metrics = metrics_m1
    best_config = config_m1
    best_model_path = m1_path

print(f"\n    >> MEJOR MODELO SELECCIONADO: {best_model_name} (ROC-AUC: {best_metrics['test_roc_auc']:.4f})")

# Exportar hiperparámetros y métricas a archivos estandarizados (DVC params & metrics)
params_nn_path = "params_nn.yaml"
with open(params_nn_path, "w", encoding="utf-8") as f:
    yaml.dump({
        "selected_model": best_model_name,
        "models": {
            "shallow_nn": config_m1,
            "deep_bn_nn": config_m2
        }
    }, f, default_flow_style=False)

metrics_nn_path = "metrics_nn.json"
with open(metrics_nn_path, "w", encoding="utf-8") as f:
    json.dump({
        "best_model": best_model_name,
        "metrics_comparison": {
            "shallow_nn": metrics_m1,
            "deep_bn_nn": metrics_m2
        }
    }, f, indent=4)

print(f"    Hiperparámetros exportados en: '{params_nn_path}'")
print(f"    Métricas exportadas en:         '{metrics_nn_path}'")


# =====================================================================
# FASE 4: ALGORITMOS DE DATA DRIFT (KS-TEST Y PSI) EN RED NEURONAL
# =====================================================================
print("\n" + "="*65)
print("[4] MONITOREO DE DATA DRIFT PARA LA RED NEURONAL")
print("="*65)

def calculate_psi(baseline, actual, num_bins=10):
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
    return float(np.sum((psi_act - psi_base) * np.log(psi_act / psi_base)))

def detect_drift_ks(baseline, actual, alpha=0.05):
    stat, p_value = ks_2samp(baseline, actual)
    return float(stat), float(p_value), bool(p_value < alpha)

# Variable monitoreada: 'Glucose'
feature_key = 'Glucose'
baseline_data = X_train[feature_key].values
mean_base = baseline_data.mean()
std_base = baseline_data.std()

# Escenario Estable
prod_stable = np.clip(np.random.normal(loc=mean_base, scale=std_base * 0.95, size=250), 40, 200)
psi_st = calculate_psi(baseline_data, prod_stable)
ks_st, p_val_st, drift_st = detect_drift_ks(baseline_data, prod_stable)

# Escenario con Drift
prod_drift = np.clip(np.random.normal(loc=mean_base + 35.0, scale=std_base * 1.15, size=250), 50, 250)
psi_dr = calculate_psi(baseline_data, prod_drift)
ks_dr, p_val_dr, drift_dr = detect_drift_ks(baseline_data, prod_drift)

print(f"--- Monitoreo de Característica: {feature_key} ---")
print(f"  Tráfico Estable:  PSI = {psi_st:.4f} | KS p-val = {p_val_st:.4f} (Drift: {drift_st}) -> OK")
print(f"  Tráfico Desviado: PSI = {psi_dr:.4f} | KS p-val = {p_val_dr:.2e} (Drift: {drift_dr}) -> ALERTA CRÍTICA")

# Generar gráfico de reporte visual
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.hist(baseline_data, bins=20, alpha=0.45, color="steelblue", label="Baseline (Entrenamiento)", density=True)
ax1.hist(prod_stable, bins=20, alpha=0.8, color="forestgreen", label="Producción Estable", density=True, histtype='step', linewidth=2.5)
ax1.set_title(f"A. Tráfico Estable - {feature_key}\nPSI: {psi_st:.4f} | KS p-val: {p_val_st:.3f}", fontsize=12, fontweight='bold')
ax1.set_xlabel("Glucosa (mg/dL)")
ax1.set_ylabel("Densidad")
ax1.legend()

ax2.hist(baseline_data, bins=20, alpha=0.45, color="steelblue", label="Baseline (Entrenamiento)", density=True)
ax2.hist(prod_drift, bins=20, alpha=0.8, color="crimson", label="Producción con Drift", density=True, histtype='step', linewidth=2.5)
ax2.set_title(f"B. Tráfico Desviado - {feature_key}\nPSI: {psi_dr:.4f} (ALERTA) | KS p-val: {p_val_dr:.2e}", fontsize=12, fontweight='bold', color="darkred")
ax2.set_xlabel("Glucosa (mg/dL)")
ax2.set_ylabel("Densidad")
ax2.legend()

plt.tight_layout()
drift_report_path = "diabetes_nn_drift_report.png"
plt.savefig(drift_report_path, dpi=150)
plt.close()
print(f"    Reporte de Drift guardado como: '{drift_report_path}'")


# =====================================================================
# FASE 5: REGISTRO DE ARTEFACTOS Y ACCESO A LA INTERFAZ W&B
# =====================================================================
print("\n" + "="*65)
print("[5] RESUMEN DE EXPERIMENTACIÓN Y ACCESO AL DASHBOARD DE W&B")
print("="*65)

if WANDB_MODE == "online":
    project_url = f"https://wandb.ai/{wandb.api.default_entity}/{WANDB_PROJECT}"
    print(f"\n[DASHBOARD]: Dashboard de Weights & Biases en la Nube:")
    print(f"   -> URL del Proyecto: {project_url}")
    print("   Abriendo la interfaz en tu navegador web...")
    try:
        webbrowser.open(project_url)
    except Exception:
        pass
else:
    print("\n[W&B LOCAL]: Weights & Biases se ejecuto en modo LOCAL / OFFLINE:")
    print("   Para sincronizar todos estos resultados y ver el dashboard interactivo en la web:")
    print("   1. Inicia sesion en terminal: wandb login")
    print(f"   2. Sincroniza los datos:     wandb sync ./wandb/")
    print(f"   3. Tu proyecto estara en:    https://wandb.ai/(tu_usuario)/{WANDB_PROJECT}")

print("\n" + "="*65)
print("RECOMENDACIÓN DVC:")
print("Versiona los nuevos modelos de Deep Learning con:")
print("   dvc add models/diabetes_nn_shallow.keras models/diabetes_nn_deep.keras models/scaler_nn.joblib")
print("   git add models/*.dvc params_nn.yaml metrics_nn.json 02_diabetes_nn_monitoreo.py")
print("   git commit -m 'feat: registrar modelos de redes neuronales con W&B y DVC'")
print("   dvc push")
print("=====================================================================\n")
