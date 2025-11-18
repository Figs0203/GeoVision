# Guía de Ejecución - GeoVision

Esta guía te llevará paso a paso para ejecutar el proyecto desde cero y entrenar un modelo de clasificación geográfica.

---

## 📋 Requisitos Previos

Antes de comenzar, asegúrate de tener:

- **Python 3.8+** instalado
- **CUDA 11.8+** (recomendado para GPU, opcional pero muy recomendado)
- **16GB+ RAM** recomendado
- **~15GB espacio en disco** para dataset, modelos y embeddings
- **GPU NVIDIA** (RTX 3050 o superior recomendado)

---

## 🔧 Instalación

### 1. Clonar el repositorio (si aplica)

```bash
git clone <url-del-repositorio>
cd GeoVision
```

### 2. Crear entorno virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Instalar PyTorch con CUDA (recomendado)

```bash
# Para CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Para CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 5. Descargar Modelo Places365

Descarga manualmente el modelo Places365:
- **URL**: http://places2.csail.mit.edu/models_places365/resnet50_places365.pth
- **Ubicación**: Coloca el archivo en `models/places365/resnet50_places365.pth`
- **Tamaño**: ~100 MB

---

## 📥 Preparación de Datos

### 1. Descargar Shards de Kaggle

Descarga los shards del dataset [Large Dataset of Geotagged Images](https://www.kaggle.com/datasets/habedi/large-dataset-of-geotagged-images) y colócalos en la carpeta `shards/`:

```
shards/
  └── shard_0.msg    (o shard_2.msg, shard_3.msg, etc.)
```

---

## 🚀 Ejecución del Pipeline

Los scripts están numerados del 0 al 7 y deben ejecutarse **en orden**. Todos tienen valores por defecto, así que puedes ejecutarlos directamente sin argumentos (algunos te guiarán con menús interactivos).

### Paso 1: Extraer Imágenes del Shard

```bash
python downloader_smart.py
```

**¿Qué hace?**
- Extrae imágenes JPEG desde archivos `.msg` (MessagePack)
- Genera `coords.csv` con coordenadas GPS (latitud, longitud) de cada imagen
- Guarda imágenes en carpeta `images/`

**Tiempo estimado:** 5-10 minutos (depende del tamaño del shard)

---

### Paso 2: Mapear Coordenadas a Continentes

```bash
python scripts/0_map_coords_to_continent.py
```

**¿Qué hace?**
- Lee `coords.csv` con coordenadas GPS
- Mapea cada coordenada a su continente usando datos geográficos
- Genera `coords_with_continent.csv` con columna adicional de continente
- **Modo incremental:** Si ejecutas de nuevo, solo procesa coordenadas nuevas

**Tiempo estimado:** 2-5 minutos

---

### Paso 3: Organizar Imágenes por Continente

```bash
python scripts/1_prepare_dataset_folders.py
```

**¿Qué hace?**
- Lee `coords_with_continent.csv`
- Organiza imágenes en estructura `data/images_by_continent/<continent>/`
- Crea carpetas: `Africa/`, `Americas/`, `Asia/`, `Europe/`, `Oceania/`
- **Modo incremental:** Solo copia imágenes nuevas (omite las que ya existen)

**Tiempo estimado:** 5-10 minutos

---

### Paso 4: Clasificar Escenas (Indoor/Outdoor)

```bash
python scripts/2_run_scene_filter.py
```

**¿Qué hace?**
- Usa Places365 ResNet-50 para clasificar cada imagen como `indoor`, `outdoor` o `unknown`
- **Filtra por confianza mínima:** Si `confidence < 0.6`, marca como `unknown`
- Filtra automáticamente todas las imágenes `unknown` (elimina ambigüedades)
- Agrega columna `continent` al CSV
- Genera `outputs/scene_predictions.csv` con: `filename`, `scene_type`, `confidence`, `continent`

**Tiempo estimado:** 15-30 minutos (depende de la GPU)

**Nota:** Este paso mejora la calidad del dataset eliminando imágenes ambiguas (ej: interiores con ventanas grandes) que Places365 clasifica incorrectamente.

---

### Paso 5: Preparar Splits Train/Test

```bash
python scripts/3_prepare_scene_dataset.py
```

**¿Qué hace?**
- Te muestra un menú para seleccionar escenas:
  - **[1] Solo outdoor** (recomendado - mejor rendimiento)
  - **[2] Solo indoor**
  - **[3] Ambos (outdoor + indoor)**
- Reorganiza imágenes en estructura `data/images_by_scene/<scene>/<continent>/`
- Genera splits persistentes: `data/train_<scene>/` y `data/test_<scene>/` (80/20)
- Calcula bins geográficos (`lat_bin`, `lon_bin`) para cada imagen
- Genera metadatos en `data/metadata/train_<scene>_metadata.csv` y `test_<scene>_metadata.csv`

**Recomendación:** Selecciona **[1] Solo outdoor** para mejores resultados.

**Tiempo estimado:** 5-10 minutos

---

### Paso 6 (Opcional): Generar Embeddings CLIP

**Este paso es opcional pero recomendado** para mejorar el rendimiento del modelo (~+5 puntos porcentuales en F1-score).

#### 6.1: Embeddings para Entrenamiento

```bash
python scripts/4_clip_compute_embeddings.py
```

Cuando te lo solicite, proporciona:
- **Metadata CSV**: `data/metadata/train_outdoor_metadata.csv`
- **Root directory**: `data/train_outdoor`
- **Output**: `data/clip_embeddings/train_outdoor_embeddings.npz`

**¿Qué hace?**
- Genera embeddings CLIP (512 dimensiones) para cada imagen de entrenamiento
- Guarda embeddings en formato `.npz` para uso posterior
- Acelera el entrenamiento (no tiene que generar embeddings en cada época)

**Tiempo estimado:** 10-20 minutos (depende de GPU y tamaño del dataset)

#### 6.2: Embeddings para Validación

```bash
python scripts/4_clip_compute_embeddings.py
```

Cuando te lo solicite, proporciona:
- **Metadata CSV**: `data/metadata/test_outdoor_metadata.csv`
- **Root directory**: `data/test_outdoor`
- **Output**: `data/clip_embeddings/test_outdoor_embeddings.npz`

**Tiempo estimado:** 3-5 minutos

---

### Paso 7: Entrenar el Modelo

```bash
python scripts/5_train_vit_trainer.py
```

**¿Qué hace?**
- Entrena un modelo Vision Transformer (ViT) con:
  - Features visuales de la imagen (ViT backbone)
  - Features de geolocalización (bins de latitud/longitud)
  - Embeddings CLIP (opcional, si completaste el Paso 6)

**Menús interactivos:**
1. **Modo de entrenamiento:**
   - **[1] Solo exteriores/interiores (scene)** - Recomendado
   - **[2] Escena + continente combinado (scene_continent)**
   
   Selecciona **[1]**.

2. **Escena a utilizar:**
   - **[1] Outdoor** - Recomendado
   - **[2] Indoor**
   
   Selecciona **[1]**.

3. **¿Integrar embeddings CLIP durante el entrenamiento?**
   - **y** (yes) - Si completaste el Paso 6
   - **N** (no) - Si no generaste embeddings CLIP
   
   Si seleccionaste "y", proporciona las rutas a los archivos `.npz` generados en el Paso 6.

**Hiperparámetros:**
- Learning rate: 2e-5
- Épocas: 12 (con early stopping, patience=3)
- Batch size: 16
- Weight decay: 0.02
- Balanceo de clases automático
- Augmentación de datos (rotación, flip, ajustes de color, ruido gaussiano)
- Detección automática de overfitting

**Tiempo estimado:** 2-4 horas con GPU (RTX 3050), 10-15 horas con CPU (no recomendado)

**Salida:**
- Checkpoints guardados en `outputs/checkpoints/vit-continent-balanced-<tag>/`
- `training_metadata.json` con toda la configuración del entrenamiento

---

### Paso 8: Evaluar el Modelo

```bash
python scripts/6_eval_metrics.py
```

**¿Qué hace?**
- Carga automáticamente el modelo más reciente entrenado
- Lee configuración desde `training_metadata.json`
- Evalúa el modelo en el conjunto de prueba
- Genera métricas completas:
  - `outputs/classification_report.txt`: Reporte detallado con precision, recall, F1-score por clase
  - `outputs/confusion_matrix.png`: Matriz de confusión visualizada

**Tiempo estimado:** 2-5 minutos

**Resultados esperados:**
- **Con CLIP**: ~75% F1-score (weighted)
- **Sin CLIP**: ~70% F1-score (weighted)

---

### Paso 9: Predecir en Imagen Nueva

```bash
python scripts/7_predict_image.py --image ruta/a/imagen.jpg
```

**Opciones adicionales:**
```bash
# Con coordenadas geográficas (mejora precisión)
python scripts/7_predict_image.py --image foto.jpg --lat 40.7128 --lon -74.0060

# Mostrar probabilidades de todas las clases
python scripts/7_predict_image.py --image foto.jpg --probs

# Mostrar imagen con visualización
python scripts/7_predict_image.py --image foto.jpg --show --probs
```

**¿Qué hace?**
- Carga automáticamente el modelo más reciente
- Clasifica la imagen en uno de los 5 continentes
- Muestra predicción y confianza
- Opcionalmente muestra probabilidades para todas las clases

---

## 📊 Resultados Esperados

Después de completar todos los pasos, deberías ver:

### Métricas Globales (con CLIP)
- **Accuracy**: ~75%
- **F1-Score**: ~75% (weighted)
- **Precision**: ~75% (weighted)
- **Recall**: ~75% (weighted)

### Performance por Continente
- **Europa**: Mejor rendimiento (~83% F1-score)
- **América**: Muy buen rendimiento (~78% F1-score)
- **Asia**: Buen rendimiento (~66% F1-score)
- **Oceanía**: Rendimiento estable (~64% F1-score)
- **África**: Menor rendimiento (~58% F1-score) - clase minoritaria

---

## ⚠️ Problemas Comunes

### "No images found to classify"
**Solución:** Asegúrate de ejecutar el Paso 3 antes del Paso 4.

### "FileNotFoundError: resnet50_places365.pth"
**Solución:** Descarga el modelo Places365 y colócalo en `models/places365/resnet50_places365.pth` (ver "Instalación" Paso 5).

### "FileNotFoundError: training_metadata.json"
**Solución:** Ejecuta el Paso 7 (entrenamiento) primero.

### Errores de memoria GPU
**Solución:** Reduce el batch size en `scripts/5_train_vit_trainer.py` (línea 152): cambia `BATCH_SIZE = 16` a `BATCH_SIZE = 8` o `4`.

### "ModuleNotFoundError: No module named 'preprocessing'"
**Solución:** Asegúrate de estar en el directorio raíz del proyecto y que `preprocessing/__init__.py` existe.

---

## 🔄 Re-ejecutar con Nuevos Datos

### Opción A: Agregar Imágenes desde Shards de Kaggle

Si quieres agregar más imágenes desde shards `.msg`:

1. **Paso 1:** Ejecuta `downloader_smart.py` (agrega nuevas imágenes automáticamente)
2. **Paso 2:** Ejecuta `scripts/0_map_coords_to_continent.py` (procesa solo nuevas coordenadas, modo incremental)
3. **Paso 3:** Ejecuta `scripts/1_prepare_dataset_folders.py` (copia solo imágenes nuevas, modo incremental)
4. **Paso 4 (Opcional):** Ejecuta `scripts/2_run_scene_filter.py` para clasificar las nuevas imágenes (solo si no son todas outdoor)
5. **Paso 5:** Ejecuta `scripts/3_prepare_scene_dataset.py` para reorganizar los splits (solo si ejecutaste el Paso 4)
6. **Pasos 6-9:** Ejecuta normalmente desde el Paso 6 si necesitas regenerar embeddings o reentrenar

**Nota:** Los Pasos 2 y 3 tienen modo incremental por defecto. Si quieres forzar recálculo completo, usa el flag `--full-refresh`:
```bash
python scripts/0_map_coords_to_continent.py --full-refresh
python scripts/1_prepare_dataset_folders.py --full-refresh
```

### Opción B: Integrar Dataset de Google Street View

Si tienes un dataset de Google Street View (todas las imágenes son outdoor):

1. **Estructura del dataset:**
   ```
   GoogleStreetViewImages/
     ├── coordsSV.csv          (columnas: lat, lon)
     └── dataset/
         ├── 0.png
         ├── 1.png
         ├── ...
         └── N.png
   ```

2. **Integrar el dataset:**
   ```bash
   python integrate_streetview.py --streetview-dir GoogleStreetViewImages --auto-mark-outdoor
   ```
   
   **¿Qué hace?**
   - Busca `coordsSV.csv` en `GoogleStreetViewImages/` o `GoogleStreetViewImages/dataset/`
   - Busca imágenes en `GoogleStreetViewImages/dataset/` o `GoogleStreetViewImages/`
   - Limpia coordenadas automáticamente (maneja formatos europeos, valores extremos, etc.)
   - Mapea coordenadas a continentes automáticamente
   - Copia y organiza imágenes en `data/images_by_continent/<continent>/`
   - Actualiza `coords.csv` y `coords_with_continent.csv` incrementalmente
   - **Marca automáticamente todas las imágenes como 'outdoor'** en `outputs/scene_predictions.csv`
   - **Modo incremental:** Solo procesa imágenes nuevas (evita duplicados por nombre de archivo)

3. **Continuar con el pipeline:**
   - **NO necesitas ejecutar** `scripts/2_run_scene_filter.py` (ya están marcadas como outdoor)
   - Ejecuta directamente `scripts/3_prepare_scene_dataset.py` para reorganizar splits
   - Continúa con los Pasos 6-9 si necesitas reentrenar

**Ventajas:**
- ✅ Integración automática e incremental
- ✅ No necesitas ejecutar el script 2 (ahorra tiempo)
- ✅ Todas las imágenes se marcan como outdoor automáticamente
- ✅ Compatible con el resto del pipeline

**Ejemplo de uso:**
```bash
# Integrar Street View
python integrate_streetview.py --streetview-dir GoogleStreetViewImages --auto-mark-outdoor

# Reorganizar splits (seleccionar [1] outdoor)
python scripts/3_prepare_scene_dataset.py

# Si quieres agregar más imágenes después (que puedan ser indoor):
python scripts/2_run_scene_filter.py  # Solo si agregas imágenes nuevas que no sean outdoor
```

---

## 📝 Notas Importantes

1. **Orden de ejecución:** Los scripts deben ejecutarse en orden (0, 1, 2, 3, 4, 5, 6, 7).

2. **Splits persistentes:** Los splits train/test se generan una vez en el Paso 5 y se reutilizan. No los modifiques manualmente.

3. **Menús interactivos:** Los scripts 3, 4 y 5 tienen menús interactivos. Solo sigue las instrucciones en pantalla.

4. **GPU recomendada:** El entrenamiento (Paso 7) es mucho más rápido con GPU. Con CPU puede tomar 10-15 horas.

5. **CLIP opcional:** El Paso 6 es opcional, pero mejora significativamente el rendimiento (+5 puntos porcentuales).

6. **Solo outdoor:** Para mejores resultados, usa solo imágenes outdoor (selecciona [1] en el Paso 5).

---

## 🎯 Resumen Rápido

### Pipeline Completo (desde cero)

```bash
# 1. Instalación (una vez)
pip install -r requirements.txt
# Descargar Places365 en models/places365/

# 2. Pipeline completo (en orden)
python downloader_smart.py                                    # Paso 1: Extraer imágenes del shard
python scripts/0_map_coords_to_continent.py                   # Paso 2: Mapear coordenadas a continentes
python scripts/1_prepare_dataset_folders.py                   # Paso 3: Organizar por continente
python scripts/2_run_scene_filter.py                          # Paso 4: Clasificar indoor/outdoor (OPCIONAL)
python scripts/3_prepare_scene_dataset.py                     # Paso 5: Reorganizar splits (seleccionar [1] outdoor)
python scripts/4_clip_compute_embeddings.py                   # Paso 6.1: Embeddings train (opcional)
python scripts/4_clip_compute_embeddings.py                   # Paso 6.2: Embeddings test (opcional)
python scripts/5_train_vit_trainer.py                         # Paso 7: Entrenar modelo (seleccionar [1], [1], y/N)
python scripts/6_eval_metrics.py                              # Paso 8: Evaluar modelo
python scripts/7_predict_image.py --image foto.jpg            # Paso 9: Predecir imagen nueva
```

### Integrar Dataset de Google Street View

```bash
# Integrar Street View (todas outdoor, no necesita script 2)
python integrate_streetview.py --streetview-dir GoogleStreetViewImages --auto-mark-outdoor
python scripts/3_prepare_scene_dataset.py                     # Reorganizar splits (seleccionar [1] outdoor)
# Continúa con Pasos 6-9 si necesitas reentrenar
```

---

¡Listo! Con estos pasos deberías poder ejecutar el proyecto completo desde cero. Si tienes problemas, revisa la sección "Problemas Comunes" o consulta el `README.md` para más detalles.

