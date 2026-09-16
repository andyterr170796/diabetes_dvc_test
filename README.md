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

> **Nota:** Si el almacenamiento remoto utiliza un proveedor cloud específico, instala el complemento correspondiente:
> - Para **Google Drive**: `pip install "dvc[gdrive]"`
> - Para **AWS S3**: `pip install "dvc[s3]"`
> - Para **Azure Blob**: `pip install "dvc[azure]"`
> - Para **Google Cloud Storage**: `pip install "dvc[gs]"`

### Paso 3: Descargar los datos con `dvc pull`
Ejecuta el siguiente comando en la raíz del proyecto:
```bash
dvc pull
```

**¿Qué ocurre internamente?**
1. DVC lee el archivo `diabetes.csv.dvc`.
2. Busca la clave hash MD5 registrada en el almacenamiento remoto configurado.
3. Descarga el archivo exacto y lo coloca en tu directorio de trabajo con su nombre original: `diabetes.csv`.

Una vez completado el comando, podrás ejecutar tus scripts normalmente (ej. `python 01_monitoreo_y_drift.py`).

---

## ⚙️ Prerrequisito: Configurar el Almacenamiento Remoto (DVC Remote)

Para que `dvc pull` funcione desde otra computadora, la persona administradora del repositorio debe haber configurado un **Remote Storage** y haber ejecutado `dvc push` previamente.

Aquí tienes los ejemplos más comunes:

### Opción A: Google Drive (Ideal para clases y pruebas)
```bash
# 1. Crear un remote apuntando al ID de una carpeta de Google Drive
dvc remote add -d myremote gdrive://<ID_DE_TU_CARPETA_DE_DRIVE>

# 2. Guardar la configuración en Git
git add .dvc/config
git commit -m "chore: configurar remote de Google Drive para DVC"
git push origin main

# 3. Subir el dataset al almacenamiento remoto
dvc push
```

### Opción B: AWS S3 / Cloud Storage
```bash
# 1. Crear el remote en S3
dvc remote add -d myremote s3://mi-bucket-mlops/dvc-storage

# 2. Subir configuración y datos
git add .dvc/config
git commit -m "chore: configurar remote S3 para DVC"
git push origin main
dvc push
```

### Opción C: Almacenamiento local o en red (Pruebas en red local)
```bash
dvc remote add -d local_storage /ruta/compartida/dvc_cache
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
