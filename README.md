# Control de Versiones de Datos (DVC) con Git y GitHub

Este repositorio es una demostración práctica de **MLOps** para el control de versiones de datos utilizando **DVC (Data Version Control)** y **Git**.

---

## 🧠 ¿Cómo funciona la arquitectura?

En proyectos de Machine Learning y Ciencia de Datos, los datasets pueden ser pesados y cambian con el tiempo. Git no está diseñado para almacenar archivos grandes ni datos binarios. 

Por ello separamos las responsabilidades:

| Herramienta | Rol | ¿Qué almacena? |
| :--- | :--- | :--- |
| **Git & GitHub** | Control de versiones de código y metadatos | Código Python, notebooks, configuraciones y los archivos puntero (`.dvc`). |
| **DVC & Remote Storage** | Control de versiones de datos grandes | Datasets (`diabetes.csv`), modelos entrenados (`.pkl`, `.onnx`), artefactos. |

```
+-------------------+                    +----------------------+
|      GitHub       |                    |  DVC Remote Storage  |
|  (Código y .dvc)  |                    | (S3 / Drive / Cloud) |
+---------+---------+                    +----------+-----------+
          ^                                         ^
          | git clone / git push                    | dvc push / dvc pull
          v                                         v
+---------------------------------------------------------------+
|                       Tu PC Local                             |
|                                                               |
|  - 01_monitoreo_y_drift.py  (Git)                             |
|  - diabetes.csv.dvc         (Git -> contiene hash MD5)        |
|  - diabetes.csv             (DVC -> ignorado en .gitignore)   |
+---------------------------------------------------------------+
```

---

## 💻 ¿Cómo obtener el archivo `diabetes.csv` si estás en otra PC?

Si clonas este repositorio en una computadora diferente, notarás que **`diabetes.csv` no existe en la carpeta**, pero sí existe el archivo puntero **`diabetes.csv.dvc`**.

Sigue estos pasos para descargarlo y reconstruirlo en la nueva máquina:

### Paso 1: Clonar el repositorio de GitHub
Abre tu terminal y clona el proyecto:
```bash
git clone https://github.com/andyterr170796/diabetes_dvc_test.git
cd diabetes_dvc_test
```

### Paso 2: Instalar DVC en tu entorno
Instala DVC en tu entorno virtual de Python:
```bash
pip install dvc
```

### Paso 3: Configurar credenciales y descargar los datos (`dvc pull`)
Como el almacenamiento remoto está alojado en **DAGsHub**, simplemente configura tus credenciales locales o usa tu token de lectura:

```bash
# 1. Autenticar con tu usuario y token de DAGsHub (solo en la máquina local):
dvc remote modify origin --local auth basic
dvc remote modify origin --local user TU_USUARIO_O_EMAIL
dvc remote modify origin --local password TU_DAGSHUB_TOKEN

# 2. Descargar los datos:
dvc pull
```

**¿Qué ocurre internamente?**
1. DVC lee el archivo `diabetes.csv.dvc`.
2. Busca la clave hash MD5 registrada en el almacenamiento remoto de DAGsHub.
3. Descarga el archivo exacto y lo coloca en tu directorio de trabajo con su nombre original: `diabetes.csv`.

Una vez completado el comando, podrás ejecutar tus scripts normalmente (ej. `python 01_monitoreo_y_drift.py`).

---

## ⚙️ Almacenamiento Remoto Configurado (DAGsHub)

El repositorio ya tiene configurado el remote oficial en **DAGsHub**:
`https://dagshub.com/andyterr170796/diabetes_dvc_test.dvc`

Para subir nuevos datos desde la PC administradora una vez autenticado:
```bash
dvc push
```

---

## 🔄 Flujo de Trabajo Cotidiano (Cheat Sheet)

| Tarea | Comando Git | Comando DVC |
| :--- | :--- | :--- |
| **Nuevo dataset o cambios en datos** | `git add diabetes.csv.dvc` | `dvc add diabetes.csv` |
| **Guardar cambios en historial** | `git commit -m "mensaje"` | Automático con `dvc add` |
| **Compartir cambios con el equipo** | `git push origin main` | `dvc push` |
| **Obtener cambios del equipo en otra PC** | `git pull origin main` | `dvc pull` |
| **Revisar estado de sincronización** | `git status` | `dvc status` |
