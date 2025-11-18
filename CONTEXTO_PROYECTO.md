# CONTEXTO COMPLETO DEL PROYECTO - GeoVision

## 📋 RESUMEN EJECUTIVO

**GeoVision** es un proyecto de clasificación de imágenes geográficas por continente usando Vision Transformer (ViT) con integración de CLIP fusionado y features de geolocalización. El modelo identifica el continente de origen (África, América, Asia, Europa, Oceanía) de una imagen con un F1-score de ~75% usando CLIP fusionado.

**Estado actual del proyecto:**
- ✅ Pipeline completo implementado y funcional
- ✅ CLIP fusionado (Opción A) implementado y funcionando
- ✅ Geolocalización (lat/lon bins) integrada
- ✅ Clasificación indoor/outdoor con Places365
- ✅ Scripts numerados del 0 al 7, ejecutables en orden
- ✅ Auto-incremental en scripts de procesamiento
- ✅ Menús interactivos para facilitar uso
- ✅ Splits persistentes train/test (evita data leakage)
- ✅ Resultados: ~75% F1-score con CLIP, ~70% sin CLIP

---

## 🎯 OBJETIVO DEL PROYECTO

Clasificar imágenes geográficas en **5 continentes**:
1. **África**
2. **América** (North + South America normalizados)
3. **Asia**
4. **Europa**
5. **Oceanía**

**Métrica principal:** F1-score (weighted average)

**Restricción:** El modelo solo predice continentes, NO escenas (indoor/outdoor). Las escenas son usadas para filtrar datos de entrenamiento pero no como clase de salida.

---

## 🏗️ ARQUITECTURA DEL MODELO

### Modelo: `GeoViTForImageClassification`

El modelo fusiona **3 fuentes de información**:

```
┌─────────────────────────────────────────────────────────┐
│  Imagen (224x224)                                       │
│  ↓                                                       │
│  ViT Backbone (google/vit-base-patch16-224-in21k)      │
│  ↓                                                       │
│  Pooled Output: [768 dimensiones]                       │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Coordenadas GPS (lat, lon)                             │
│  ↓                                                       │
│  Binning: lat → 18 bins (10° cada uno),                │
│           lon → 36 bins (10° cada uno)                  │
│  ↓                                                       │
│  Embedding Layers:                                      │
│    - lat_embedding: Embedding(19, 32)                  │
│    - lon_embedding: Embedding(37, 32)                  │
│  ↓                                                       │
│  Geo Features: [32 + 32 = 64 dimensiones]               │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Embedding CLIP (512 dim) - OPCIONAL                   │
│  ↓                                                       │
│  Proyección: Linear(512 → 128)                         │
│    + LayerNorm                                          │
│    + GELU                                               │
│    + Dropout                                            │
│  ↓                                                       │
│  CLIP Features: [128 dimensiones]                       │
└─────────────────────────────────────────────────────────┘

                        ↓
        ┌───────────────────────────────┐
        │   CONCATENACIÓN               │
        │   768 + 64 + 128 = 960 dim    │
        └───────────────────────────────┘
                        ↓
                Dropout (0.1)
                        ↓
        ┌───────────────────────────────┐
        │   Classifier (Linear)         │
        │   960 → 5 clases              │
        └───────────────────────────────┘
                        ↓
            Logits (5 clases)
```

### Componentes Clave:

1. **Backbone ViT**: 
   - Modelo: `google/vit-base-patch16-224-in21k`
   - Hidden size: 768 dimensiones
   - Pooled output usado como features visuales principales

2. **Geolocalización**:
   - **LAT_BIN_SIZE = 10°**: Divide latitud [-90, 90] en 18 bins de 10° cada uno
   - **LON_BIN_SIZE = 10°**: Divide longitud [-180, 180] en 36 bins de 10° cada uno
   - **UNKNOWN_BIN = -1**: Para coordenadas faltantes (NaN)
   - **Embedding dim = 32**: Cada bin (lat/lon) se convierte en vector de 32 dim
   - **Total bins**: LAT_BINS_TOTAL = 19 (18 bins + 1 para unknown), LON_BINS_TOTAL = 37 (36 bins + 1 para unknown)

3. **CLIP Fusionado (Opción A)**:
   - **Input**: Embeddings CLIP de 512 dimensiones (del modelo `openai/clip-vit-base-patch32`)
   - **Proyección**: Linear → LayerNorm → GELU → Dropout
   - **Output**: 128 dimensiones
   - **Integración**: Concatenación directa con ViT + geo features
   - **Entrenamiento**: End-to-end (CLIP se proyecta, ViT se ajusta)

4. **Clasificador Final**:
   - Input: 960 dimensiones (768 ViT + 64 geo + 128 CLIP) o 832 sin CLIP
   - Output: 5 logits (una por cada continente)

---

## 📁 ESTRUCTURA COMPLETA DEL PROYECTO

### Estructura de Directorios

```
GeoVision/
│
├── 📄 downloader_smart.py              # Extracción inteligente de imágenes del shard
│   ├── Modo incremental (continúa desde último índice)
│   ├── Filtrado opcional por continente
│   └── Genera: coords.csv
│
├── 📁 scripts/                         # Scripts principales numerados 0-7
│   ├── 0_map_coords_to_continent.py    # Mapeo GPS → continente (incremental)
│   ├── 1_prepare_dataset_folders.py    # Organización por continente (incremental)
│   ├── 2_run_scene_filter.py           # Clasificación indoor/outdoor (Places365)
│   ├── 3_prepare_scene_dataset.py      # Reorganización scene/continent + splits
│   ├── 4_clip_compute_embeddings.py    # Generación embeddings CLIP (opcional)
│   ├── 5_train_vit_trainer.py          # Entrenamiento ViT + CLIP fusionado
│   ├── 6_eval_metrics.py               # Evaluación y métricas
│   └── 7_predict_image.py              # Predicción en imágenes nuevas
│
├── 📁 preprocessing/                   # Módulos de preprocesamiento
│   ├── __init__.py                     # Hace que sea un paquete Python
│   ├── scene_classifier.py             # Clasificador Places365 (indoor/outdoor)
│   └── geo_dataset.py                  # Dataset PyTorch con geo features y CLIP
│
├── 📁 models/                          # Modelos pre-entrenados
│   └── places365/
│       ├── resnet50_places365.pth      # Modelo ResNet-50 Places365 (DESCARGAR)
│       ├── categories_places365.txt    # Categorías Places365 (incluido)
│       └── IO_places365.txt            # Split indoor/outdoor (incluido)
│
├── 📁 data/                            # Datos del proyecto
│   ├── images_by_continent/            # Estructura inicial: <continent>/
│   ├── images_by_scene/                # Estructura final: <scene>/<continent>/
│   ├── train_outdoor/                  # Split entrenamiento (persistente)
│   ├── test_outdoor/                   # Split prueba (persistente)
│   ├── metadata/                       # CSVs con metadatos
│   │   ├── train_outdoor_metadata.csv  # Metadata train (con lat_bin, lon_bin)
│   │   └── test_outdoor_metadata.csv   # Metadata test (con lat_bin, lon_bin)
│   └── clip_embeddings/                # Embeddings CLIP pre-computados
│       ├── train_outdoor_embeddings.npz
│       └── test_outdoor_embeddings.npz
│
├── 📁 outputs/                         # Resultados y modelos entrenados
│   ├── checkpoints/
│   │   └── vit-continent-balanced-outdoor/
│   │       ├── checkpoint-XXXX/
│   │       │   ├── model.safetensors
│   │       │   └── ...
│   │       └── training_metadata.json  # Configuración completa del entrenamiento
│   ├── confusion_matrix.png            # Matriz de confusión visualizada
│   ├── classification_report.txt       # Reporte detallado de métricas
│   └── scene_predictions.csv           # Clasificaciones indoor/outdoor
│
├── 📁 shards/                          # Shards MessagePack (descargar de Kaggle)
│   └── shard_X.msg                     # Archivos .msg con imágenes
│
├── 📁 images/                          # Imágenes extraídas (generado)
│   └── shard0_XXXXX.jpg                # Formato: shard0_00000.jpg
│
├── 📄 coords.csv                       # Coordenadas extraídas (generado)
│   └── Columnas: filename, lat, lon, index
│
├── 📄 coords_with_continent.csv        # Coordenadas + continente (generado)
│   └── Columnas: filename, lat, lon, continent, index
│
├── 📄 requirements.txt                 # Dependencias Python
├── 📄 README.md                        # Documentación principal
├── 📄 GUIA_EJECUCION.md                # Guía paso a paso detallada
└── 📄 CONTEXTO_PROYECTO.md             # Este archivo (contexto completo)
```

---

## 🔄 FLUJO COMPLETO DE EJECUCIÓN

### Orden de Ejecución (Scripts 0-7)

```
┌─────────────────────────────────────────────────────────┐
│  1. downloader_smart.py                                 │
│     Extrae imágenes desde shards .msg                   │
│     ↓                                                    │
│     Genera: coords.csv                                  │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  2. scripts/0_map_coords_to_continent.py                │
│     Mapea coordenadas GPS a continentes                 │
│     - Modo incremental (no recalcula si existe CSV)     │
│     - Soporta --full-refresh                            │
│     ↓                                                    │
│     Genera: coords_with_continent.csv                   │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  3. scripts/1_prepare_dataset_folders.py                │
│     Organiza imágenes por continente                    │
│     - Modo incremental (skip si archivo existe)         │
│     - Soporta --full-refresh                            │
│     ↓                                                    │
│     Genera: data/images_by_continent/<continent>/       │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  4. scripts/2_run_scene_filter.py                       │
│     Clasifica indoor/outdoor usando Places365           │
│     - Filtra automáticamente 'unknown'                  │
│     - Agrega columna 'continent' al CSV                 │
│     ↓                                                    │
│     Genera: outputs/scene_predictions.csv               │
│     Columnas: filename, scene_type, confidence, continent│
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  5. scripts/3_prepare_scene_dataset.py                  │
│     Reorganiza por escena + continente                  │
│     - Menú interactivo: [1] outdoor, [2] indoor, [3] ambos│
│     - Genera splits persistentes train/test             │
│     - Crea metadatos con lat_bin y lon_bin              │
│     ↓                                                    │
│     Genera:                                              │
│     - data/images_by_scene/<scene>/<continent>/         │
│     - data/train_<scene>/ y data/test_<scene>/          │
│     - data/metadata/train_<scene>_metadata.csv          │
│     - data/metadata/test_<scene>_metadata.csv           │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  6. scripts/4_clip_compute_embeddings.py (OPCIONAL)     │
│     Genera embeddings CLIP                              │
│     - Menú interactivo para rutas                       │
│     - Guarda modelo CLIP usado                          │
│     ↓                                                    │
│     Ejecutar 2 veces:                                   │
│     - Para train: data/metadata/train_outdoor_metadata.csv│
│     - Para test: data/metadata/test_outdoor_metadata.csv │
│     ↓                                                    │
│     Genera:                                              │
│     - data/clip_embeddings/train_outdoor_embeddings.npz │
│     - data/clip_embeddings/test_outdoor_embeddings.npz  │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  7. scripts/5_train_vit_trainer.py                      │
│     Entrena modelo ViT con CLIP fusionado               │
│     - Menú 1: Modo entrenamiento [1] scene [2] scene_continent│
│     - Menú 2: Escena [1] outdoor [2] indoor            │
│     - Menú 3: ¿Integrar CLIP? (y/N)                     │
│     - Si CLIP: rutas a embeddings .npz                  │
│     ↓                                                    │
│     Entrena: GeoViTForImageClassification               │
│     - Hiperparámetros: LR=3e-5, epochs=12, batch=16     │
│     - Early stopping (patience=5)                       │
│     - WeightedRandomSampler para balanceo               │
│     ↓                                                    │
│     Genera:                                              │
│     - outputs/checkpoints/vit-continent-balanced-<tag>/ │
│     - training_metadata.json (configuración completa)   │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  8. scripts/6_eval_metrics.py                           │
│     Evalúa modelo y genera métricas                     │
│     - Lee training_metadata.json automáticamente        │
│     - Carga mejor checkpoint automáticamente            │
│     - Colapsa scene-continent en solo continente        │
│     ↓                                                    │
│     Genera:                                              │
│     - outputs/classification_report.txt                 │
│     - outputs/confusion_matrix.png                      │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  9. scripts/7_predict_image.py                          │
│     Predice continente de imagen nueva                  │
│     - Carga modelo desde metadata automáticamente       │
│     - Soporta --lat y --lon opcionales                  │
│     - Genera CLIP embedding on-the-fly si fue entrenado │
│     ↓                                                    │
│     Muestra: predicción + probabilidades                │
└─────────────────────────────────────────────────────────┘
```

---

## 🧩 DESCRIPCIÓN DETALLADA DE CADA SCRIPT

### `downloader_smart.py`

**Propósito:** Extraer imágenes desde archivos `.msg` (MessagePack) con soporte incremental.

**Características:**
- Detecta índice inicial desde archivos existentes en `images/`
- Carga CSV existente (`coords.csv`) si existe
- Continúa desde último índice procesado
- Filtrado opcional por continente (variable `FILTER_CONTINENTS`)
- Normaliza nombres de archivos: `shard0_00000.jpg`

**Configuración:**
```python
SHARD_FILE = "./shards/shard_3.msg"
SAVE_DIR = "./images"
CSV_FILE = "coords.csv"
FILTER_CONTINENTS = None  # o ['Asia'], ['Europe'], etc.
```

**Salida:**
- `images/` con imágenes numeradas
- `coords.csv` con columnas: `filename`, `lat`, `lon`, `index`

---

### `scripts/0_map_coords_to_continent.py`

**Propósito:** Mapear coordenadas GPS a nombres de continentes usando `geopandas` y `cartopy`.

**Características:**
- **Modo incremental**: Detecta CSV previo (`coords_with_continent.csv`) y solo procesa nuevas filas
- **Normalización**: "North America" y "South America" → "Americas"
- Usa shapefile de Cartopy (`natural_earth`) para spatial join
- Soporta `--full-refresh` para recalcular todo

**Argumentos:**
- `--csv-in`: Default `coords.csv`
- `--csv-out`: Default `coords_with_continent.csv`
- `--full-refresh`: Forzar recálculo completo

**Salida:**
- `coords_with_continent.csv` con columna adicional: `continent`

**Algoritmo:**
1. Carga CSV existente si existe (modo incremental)
2. Identifica filas nuevas vs ya mapeadas
3. Spatial join con shapefile de Cartopy para nuevas filas
4. Combina resultados y guarda

---

### `scripts/1_prepare_dataset_folders.py`

**Propósito:** Organizar imágenes en estructura `data/images_by_continent/<continent>/`.

**Características:**
- **Modo incremental**: Solo copia imágenes que no existen en destino
- **Skip inteligente**: Si `dst_path.exists()` y no `--full-refresh`, omite
- Normaliza continentes (North/South America → Americas)
- Filtra "Unknown"
- Opción `--create-split` para generar train/test (opcional, no usado normalmente)

**Argumentos:**
- `--csv`: Default `coords_with_continent.csv`
- `--source-dir`: Default `images`
- `--dest-dir`: Default `data/images_by_continent`
- `--full-refresh`: Eliminar destino y copiar todo

**Salida:**
- `data/images_by_continent/Africa/`
- `data/images_by_continent/Americas/`
- `data/images_by_continent/Asia/`
- `data/images_by_continent/Europe/`
- `data/images_by_continent/Oceania/`

**Métricas:**
- Reporta: copiadas, omitidas (ya existían), total

---

### `scripts/2_run_scene_filter.py`

**Propósito:** Clasificar imágenes como indoor/outdoor/unknown usando Places365 ResNet-50.

**Características:**
- Usa modelo Places365 pre-entrenado (`models/places365/resnet50_places365.pth`)
- Clasificación por lotes (batch_size=32)
- **Filtrado por confianza mínima**: Si `confidence < 0.6` (configurable), marca como `unknown`
- Filtrado automático de `unknown` (incluye las de baja confianza)
- Agrega columna `continent` desde `coords_with_continent.csv`
- Normaliza rutas (relativas)

**Argumentos:**
- `--image-dir`: Default `data/images_by_continent`
- `--mapping-csv`: Default `coords_with_continent.csv`
- `--output`: Default `outputs/scene_predictions.csv`
- `--min-confidence`: Default `0.6` - Confianza mínima para aceptar clasificación. Si es menor, se marca como `unknown`

**Salida:**
- `outputs/scene_predictions.csv` con columnas:
  - `filename`: Nombre del archivo
  - `scene_type`: "indoor" o "outdoor"
  - `confidence`: Confianza de la clasificación
  - `continent`: Continente de la imagen

**Algoritmo:**
1. Carga mapeo filename → continent desde CSV
2. Encuentra todas las imágenes en `image_dir`
3. Clasifica en lotes usando Places365
4. **Filtra por confianza mínima**: Marca como `unknown` si `confidence < min_confidence` (default 0.6)
5. Filtra todas las `unknown` (incluye originales y las de baja confianza)
6. Agrega `continent` y guarda CSV

**Nota importante:** El filtro de confianza mínima ayuda a eliminar imágenes ambiguas (ej: interiores con ventanas grandes) que Places365 clasifica incorrectamente, mejorando la calidad del dataset y subiendo el F1-score antes incluso del entrenamiento.

---

### `scripts/3_prepare_scene_dataset.py`

**Propósito:** Reorganizar dataset por escena + continente y generar splits persistentes train/test.

**Características:**
- **Menú interactivo**: [1] Solo outdoor, [2] Solo indoor, [3] Ambos
- **Splits persistentes**: Genera `data/train_<scene>/` y `data/test_<scene>/` (80/20)
- **Sin solapamiento**: Splits se generan una vez y se reutilizan
- **Metadatos con bins**: Calcula `lat_bin` y `lon_bin` para cada imagen
- Limpia destinos por defecto (`--clean-dest`, `--clean-splits`)

**Argumentos:**
- `--csv`: Default `outputs/scene_predictions.csv`
- `--coords-csv`: Default `coords_with_continent.csv` (para lat/lon)
- `--scenes`: Puede pasarse como argumento o se solicita por menú
- `--source-dir`: Default `data/images_by_continent`
- `--dest-dir`: Default `data/images_by_scene`
- `--split-base`: Default `data`
- `--split-size`: Default 0.2 (20% test)
- `--split-seed`: Default 42 (para reproducibilidad)

**Salida:**
- `data/images_by_scene/outdoor/Africa/`, `outdoor/Americas/`, etc.
- `data/train_outdoor/<continent>/` y `data/test_outdoor/<continent>/`
- `data/metadata/train_outdoor_metadata.csv` con columnas:
  - `relative_path`, `label_name`, `scene_type`, `continent`, `lat_bin`, `lon_bin`, `lat`, `lon`
- `data/metadata/test_outdoor_metadata.csv` (mismo formato)

**Algoritmo:**
1. Menú para seleccionar escenas
2. Filtra CSV por escenas seleccionadas
3. Calcula `lat_bin` y `lon_bin` usando funciones de `preprocessing/geo_dataset.py`
4. Genera splits estratificados por continente (80/20)
5. Copia imágenes a estructura jerárquica y splits
6. Genera CSVs de metadata con todos los features necesarios

---

### `scripts/4_clip_compute_embeddings.py` (OPCIONAL)

**Propósito:** Pre-computar embeddings CLIP para acelerar entrenamiento y mejorar precisión.

**Características:**
- Genera embeddings de 512 dimensiones usando `openai/clip-vit-base-patch32`
- Normaliza embeddings (L2 normalization)
- Guarda modelo CLIP usado en `.npz`
- Menú interactivo si faltan argumentos
- Procesamiento por lotes (batch_size=64 por defecto)

**Argumentos:**
- `--metadata`: CSV con columna `relative_path`
- `--root-dir`: Directorio raíz donde están las imágenes
- `--output`: Archivo `.npz` de salida
- `--batch-size`: Default 64
- `--model-name`: Default `openai/clip-vit-base-patch32`

**Salida:**
- Archivo `.npz` con:
  - `embeddings`: Array numpy de shape (N, 512)
  - `paths`: Array de rutas relativas (strings)
  - `model_name`: Nombre del modelo CLIP usado

**Uso típico:**
1. Primera ejecución: Para train
   - Metadata: `data/metadata/train_outdoor_metadata.csv`
   - Root dir: `data/train_outdoor`
   - Output: `data/clip_embeddings/train_outdoor_embeddings.npz`

2. Segunda ejecución: Para test
   - Metadata: `data/metadata/test_outdoor_metadata.csv`
   - Root dir: `data/test_outdoor`
   - Output: `data/clip_embeddings/test_outdoor_embeddings.npz`

---

### `scripts/5_train_vit_trainer.py` ⭐ (SCRIPT PRINCIPAL)

**Propósito:** Entrenar modelo ViT con geolocalización y opcionalmente CLIP fusionado.

**Características:**
- **Menús interactivos**:
  1. Modo entrenamiento: [1] scene (solo outdoor/indoor), [2] scene_continent (ambos combinados)
  2. Escena (si modo 1): [1] outdoor, [2] indoor
  3. ¿Integrar CLIP?: (y/N) con prompts para rutas de embeddings
- **Modelo**: `GeoViTForImageClassification` con:
  - Backbone ViT (768 dim)
  - Embeddings geo (lat/lon bins, 32 dim cada uno)
  - Proyección CLIP (128 dim, si está habilitado)
- **Hiperparámetros**:
  - Learning rate: 3e-5
  - Weight decay: 0.01
  - Batch size: 16
  - Épocas: 12 (con early stopping, patience=5)
  - Scheduler: cosine con warmup_ratio=0.1
  - Eval/save steps: 500
  - Save total limit: 2 (mantiene solo 2 checkpoints)
  - FP16: Habilitado si CUDA disponible
- **Balanceo de clases**: WeightedRandomSampler
- **Augmentación**: Albumentations (RandomResizedCrop, HorizontalFlip, ColorJitter, Rotate)
- **Early Stopping**: Basado en F1-score (patience=5, threshold=0.001)
- **Métrica principal**: F1-score (weighted)

**Argumentos (todos opcionales, se solicitan por menú):**
- `--training-mode`: "scene" o "scene_continent"
- `--scene-type`: "outdoor" o "indoor"
- `--clip-train-embeddings`: Path a `.npz` de train
- `--clip-test-embeddings`: Path a `.npz` de test
- `--clip-projection-dim`: Default 128
- `--clip-model-name`: Identificador del modelo CLIP

**Configuración de rutas (según modo):**
- Modo "scene" + "outdoor":
  - Train: `data/train_outdoor`
  - Test: `data/test_outdoor`
  - Metadata train: `data/metadata/train_outdoor_metadata.csv`
  - Metadata test: `data/metadata/test_outdoor_metadata.csv`
  - Output: `outputs/checkpoints/vit-continent-balanced-outdoor`

- Modo "scene_continent":
  - Train: `data/train_scene_continent`
  - Test: `data/test_scene_continent`
  - Metadata train: `data/metadata/train_scene_continent_metadata.csv`
  - Metadata test: `data/metadata/test_scene_continent_metadata.csv`
  - Output: `outputs/checkpoints/vit-continent-balanced-scene-continent`

**Salida:**
- Checkpoints en `outputs/checkpoints/vit-continent-balanced-<tag>/`
  - Subdirectorios: `checkpoint-500/`, `checkpoint-1000/`, etc.
  - Mejor checkpoint: `checkpoint-XXXX/` (según F1-score)
  - Archivos por checkpoint: `model.safetensors` o `pytorch_model.bin`
- `training_metadata.json` en directorio raíz del modelo:
  ```json
  {
    "training_mode": "scene",
    "scene_type": "outdoor",
    "train_dir": "...",
    "test_dir": "...",
    "train_metadata": "...",
    "test_metadata": "...",
    "label_names": ["Africa", "Americas", "Asia", "Europe", "Oceania"],
    "geo_embed_dim": 32,
    "model_name": "google/vit-base-patch16-224-in21k",
    "num_lat_bins": 19,
    "num_lon_bins": 37,
    "use_clip_embeddings": true,
    "clip_train_embeddings": "...",
    "clip_test_embeddings": "...",
    "clip_input_dim": 512,
    "clip_projection_dim": 128,
    "clip_model_name": "openai/clip-vit-base-patch32",
    "checkpoint_dir": "checkpoint-8000"
  }
  ```

**Algoritmo de entrenamiento:**
1. Carga metadatos train/test
2. Carga embeddings CLIP si está habilitado
3. Crea `GeoImageDataset` con clip_embeddings
4. Construye modelo `GeoViTForImageClassification`
5. Calcula pesos para balanceo de clases
6. Crea `WeightedRandomSampler`
7. Configura `TrainingArguments` (Hugging Face)
8. Crea `BalancedTrainer` (custom Trainer con sampler balanceado)
9. Entrena con early stopping
10. Guarda mejor checkpoint y metadata

---

### `scripts/6_eval_metrics.py`

**Propósito:** Evaluar modelo entrenado y generar métricas completas.

**Características:**
- **Carga automática**: Lee `training_metadata.json` del modelo más reciente o especificado
- **Checkpoint automático**: Encuentra mejor checkpoint automáticamente
- **Soporte para modelos con/sin CLIP**: Detecta desde metadata
- **Colapso automático**: Colapsa etiquetas "outdoor-Africa" → "Africa" por defecto
- Genera matriz de confusión visualizada
- Genera reporte de clasificación completo

**Argumentos:**
- `--model-dir`: Directorio del modelo (default: más reciente con metadata)
- `--no-collapse-scene`: No colapsar etiquetas scene-continent (para modo scene_continent)

**Algoritmo:**
1. Encuentra modelo más reciente o usa `--model-dir`
2. Lee `training_metadata.json`
3. Carga embeddings CLIP si `use_clip_embeddings=true`
4. Carga mejor checkpoint (desde `checkpoint_dir` en metadata o busca `checkpoint-*/`)
5. Carga modelo `GeoViTForImageClassification` con configuración correcta
6. Evalúa en test set
7. Colapsa etiquetas (opcional)
8. Genera métricas: accuracy, precision, recall, F1-score
9. Genera matriz de confusión (seaborn heatmap)
10. Guarda reporte en `outputs/classification_report.txt`

**Salida:**
- `outputs/classification_report.txt`: Reporte detallado con métricas por clase
- `outputs/confusion_matrix.png`: Matriz de confusión visualizada

---

### `scripts/7_predict_image.py`

**Propósito:** Predecir continente de una imagen nueva.

**Características:**
- **Carga automática**: Lee `training_metadata.json` del modelo más reciente
- **Coordenadas opcionales**: Soporta `--lat` y `--lon` para mejorar precisión
- **CLIP on-the-fly**: Genera embedding CLIP dinámicamente si el modelo fue entrenado con CLIP
- Visualización opcional de imagen y probabilidades

**Argumentos:**
- `--image`: Ruta a imagen (requerido)
- `--model`: Directorio del modelo (default: más reciente)
- `--lat`: Latitud opcional
- `--lon`: Longitud opcional
- `--show`: Mostrar imagen con visualización
- `--probs`: Mostrar probabilidades de todas las clases

**Algoritmo:**
1. Lee `training_metadata.json`
2. Carga mejor checkpoint
3. Carga modelo `GeoViTForImageClassification`
4. Procesa imagen con `AutoImageProcessor`
5. Calcula `lat_bin` y `lon_bin` (si se proporcionaron coordenadas, sino usa UNKNOWN_BIN)
6. Si modelo fue entrenado con CLIP:
   - Carga modelo CLIP (`openai/clip-vit-base-patch32`)
   - Genera embedding de la imagen
   - Normaliza embedding
7. Ejecuta forward pass del modelo
8. Muestra predicción y probabilidades

---

## 📦 MÓDULOS DE PREPROCESAMIENTO

### `preprocessing/geo_dataset.py`

**Constantes geográficas:**
```python
LAT_BIN_SIZE = 10  # grados
LON_BIN_SIZE = 10  # grados
NUM_LAT_BINS = 18  # [-90, 90) dividido en 18 bins de 10°
NUM_LON_BINS = 36  # [-180, 180) dividido en 36 bins de 10°
UNKNOWN_BIN = -1   # Para coordenadas faltantes
```

**Funciones:**
- `compute_lat_bin(lat: float) -> int`: Calcula bin de latitud
- `compute_lon_bin(lon: float) -> int`: Calcula bin de longitud
- `load_clip_embeddings(npz_path: Path) -> Tuple[Dict[str, np.ndarray], int, Optional[str]]`: Carga embeddings desde `.npz`

**Clase: `GeoImageDataset`**

Dataset PyTorch que carga imágenes junto con:
- Labels (continente)
- Lat/lon bins (para embeddings geográficos)
- CLIP embeddings opcionales (para CLIP fusionado)

**Constructor:**
```python
GeoImageDataset(
    metadata: pd.DataFrame,           # CSV con metadatos
    root_dir: Path,                   # Directorio raíz de imágenes
    transforms,                       # Albumentations transforms
    label_to_id: Dict[str, int],      # Mapeo label → id
    lat_bins_total: int,              # Total bins lat (19)
    lon_bins_total: int,              # Total bins lon (37)
    clip_embeddings: Optional[Dict[str, np.ndarray]] = None,  # Mapeo path → embedding
    clip_dim: Optional[int] = None,   # Dimensión de embeddings CLIP (512)
)
```

**Retorna (__getitem__):**
```python
{
    "pixel_values": torch.Tensor,      # Imagen procesada (3, 224, 224)
    "labels": int,                     # ID de la clase
    "lat_bins": int,                   # Índice del bin de latitud
    "lon_bins": int,                   # Índice del bin de longitud
    "clip_embeddings": torch.Tensor,   # Embedding CLIP (512 dim) - opcional
}
```

---

### `preprocessing/scene_classifier.py`

**Clase: `Places365SceneClassifier`**

Clasificador indoor/outdoor usando Places365 ResNet-50.

**Constructor:**
```python
Places365SceneClassifier(
    model_path: Path,                 # Path a resnet50_places365.pth
    categories_path: Path,            # Path a categories_places365.txt
    io_path: Path,                    # Path a IO_places365.txt
    device: Optional[str] = None,     # "cuda" o "cpu"
    batch_size: int = 32,
)
```

**Método principal:**
```python
classify_paths(
    image_paths: Sequence[Path],
    threshold: float = 0.1,
) -> List[dict]
```

**Retorna:**
```python
[
    {
        "filename": str,              # Path completo del archivo
        "scene_type": str,            # "indoor", "outdoor", o "unknown"
        "indoor_confidence": float,   # Confianza indoor
        "outdoor_confidence": float,  # Confianza outdoor
        "confidence": float,          # Confianza final
    },
    ...
]
```

**Algoritmo:**
1. Carga modelo ResNet-50 y pesos Places365
2. Carga índices de categorías indoor/outdoor desde `IO_places365.txt`
3. Para cada batch de imágenes:
   - Preprocesa con transforms Places365
   - Obtiene logits del modelo
   - Suma probabilidades de categorías indoor
   - Suma probabilidades de categorías outdoor
   - Decide label basado en diferencia (threshold=0.1)

---

## ⚙️ CONFIGURACIONES Y PARÁMETROS

### Hiperparámetros del Entrenamiento

```python
# Modelo base
MODEL_NAME = 'google/vit-base-patch16-224-in21k'
HIDDEN_SIZE = 768  # Output del backbone ViT

# Geolocalización
GEO_EMBED_DIM = 32              # Dimensión de embeddings lat/lon
LAT_BIN_SIZE = 10               # Grados por bin de latitud
LON_BIN_SIZE = 10               # Grados por bin de longitud
NUM_LAT_BINS = 18               # Bins de latitud
NUM_LON_BINS = 36               # Bins de longitud
LAT_BINS_TOTAL = 19             # 18 bins + 1 para unknown
LON_BINS_TOTAL = 37             # 36 bins + 1 para unknown

# CLIP
CLIP_MODEL_NAME = 'openai/clip-vit-base-patch32'
CLIP_INPUT_DIM = 512            # Dimensión de embeddings CLIP originales
CLIP_PROJECTION_DIM = 128       # Dimensión después de proyección

# Entrenamiento
BATCH_SIZE = 16
NUM_EPOCHS = 12
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
LR_SCHEDULER = "cosine"
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_THRESHOLD = 0.001
EVAL_STEPS = 500
SAVE_STEPS = 500
SAVE_TOTAL_LIMIT = 2
METRIC_FOR_BEST_MODEL = "f1"    # F1-score (weighted)
FP16 = True                     # Si CUDA disponible

# Augmentación (Albumentations)
TRAIN_TRANSFORMS:
  - RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0))
  - HorizontalFlip(p=0.5)
  - ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5)
  - Rotate(limit=15, p=0.5)
  - Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

TEST_TRANSFORMS:
  - Resize(height=224, width=224)
  - Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
```

---

## 📊 ESTRUCTURA DE DATOS

### CSVs Generados

#### `coords.csv`
```csv
filename,lat,lon,index
shard0_00000.jpg,40.7128,-74.0060,0
shard0_00001.jpg,34.0522,-118.2437,1
...
```

#### `coords_with_continent.csv`
```csv
filename,lat,lon,continent,index
shard0_00000.jpg,40.7128,-74.0060,Americas,0
shard0_00001.jpg,34.0522,-118.2437,Americas,1
...
```

#### `outputs/scene_predictions.csv`
```csv
filename,scene_type,confidence,continent
shard0_00000.jpg,outdoor,0.95,Americas
shard0_00001.jpg,indoor,0.87,Americas
...
```

#### `data/metadata/train_outdoor_metadata.csv`
```csv
relative_path,label_name,scene_type,continent,lat_bin,lon_bin,lat,lon
Africa/IMG_001.jpg,Africa,outdoor,Africa,9,18,-5.0,20.0
Americas/IMG_002.jpg,Americas,outdoor,Americas,13,25,40.0,-80.0
...
```

#### `data/metadata/test_outdoor_metadata.csv`
```csv
# Mismo formato que train_outdoor_metadata.csv
```

---

## 🔧 DECISIONES TÉCNICAS IMPORTANTES

### 1. Separación Indoor/Outdoor

**Decisión:** Entrenar solo con imágenes **outdoor** por defecto.

**Razón:**
- Imágenes indoor tienen mucho ruido y son difíciles de clasificar geográficamente
- Mejora significativa en rendimiento al excluir indoor
- El modelo alcanza ~75% F1-score con outdoor, ~52% con ambos combinados

**Implementación:**
- Script 3 tiene menú para seleccionar [1] outdoor, [2] indoor, [3] ambos
- **Recomendación**: Usar siempre [1] outdoor

---

### 2. CLIP: Fusión (Opción A) vs Ensemble (Opción B)

**Decisión:** Implementada solo **Opción A (Fusión)**.

**Opción A (Fusión - IMPLEMENTADA):**
- CLIP se integra directamente en el modelo durante entrenamiento
- Proyección: Linear(512→128) + LayerNorm + GELU + Dropout
- Entrenamiento end-to-end
- Mejor rendimiento: ~75% F1-score
- **Ventaja**: Un solo modelo, mejor generalización

**Opción B (Ensemble - ELIMINADA):**
- Clasificador CLIP separado (Logistic Regression)
- Promedio de probabilidades durante evaluación
- Requiere dos modelos
- **Razón de eliminación**: Opción A es mejor y más simple

**Código eliminado:**
- `scripts/train_clip_classifier.py` (eliminado)
- Código de ensemble en `scripts/6_eval_metrics.py` (eliminado)

---

### 3. Splits Persistentes

**Decisión:** Splits train/test se generan **una sola vez** y se reutilizan.

**Implementación:**
- Script 3 genera splits en `data/train_<scene>/` y `data/test_<scene>/`
- Estos splits son **persistentes** (no se regeneran)
- Script 5 lee desde estos splits directamente
- **Ventaja**: Evita data leakage, garantiza reproducibilidad

**Semilla:** 42 (fija para reproducibilidad)

---

### 4. Auto-Incremental

**Decisión:** Scripts de procesamiento tienen modo incremental para no reprocesar datos.

**Implementado en:**
- `scripts/0_map_coords_to_continent.py`: Solo mapea nuevas coordenadas
- `scripts/1_prepare_dataset_folders.py`: Solo copia imágenes nuevas

**Cómo funciona:**
- Script 0: Detecta `coords_with_continent.csv` y solo procesa filas nuevas
- Script 1: Detecta si archivo destino existe y lo omite (skip)

**Forzar recálculo:**
- Usar flag `--full-refresh` en ambos scripts

---

### 5. Balanceo de Clases

**Implementación:** `WeightedRandomSampler` de PyTorch.

**Cálculo de pesos:**
```python
# Para cada clase
weight[clase] = total_imágenes / (num_clases * count[clase])

# Ejemplo con 5 clases:
# Si clase A tiene 1000 imágenes de 10000 totales:
# weight[A] = 10000 / (5 * 1000) = 2.0
# Si clase B tiene 5000 imágenes:
# weight[B] = 10000 / (5 * 5000) = 0.4
```

**Uso:**
- `WeightedRandomSampler` en `BalancedTrainer.get_train_dataloader()`
- Cada época, clases minoritarias se muestrean más frecuentemente

---

### 6. Early Stopping

**Configuración:**
- Patience: 5 evaluaciones sin mejora
- Threshold: 0.001 (mejora mínima para considerar como mejora)
- Métrica: F1-score (weighted)
- **Importante**: Solo detiene si no hay mejora por 5 evaluaciones consecutivas

**Comportamiento:**
- Evalúa cada 500 steps
- Guarda mejor checkpoint según F1-score
- Si no mejora por 5 evaluaciones (2500 steps), detiene entrenamiento
- Restaura mejor checkpoint al final

---

### 7. Menús Interactivos

**Decisión:** Scripts clave tienen menús para facilitar uso sin argumentos.

**Scripts con menús:**
- `scripts/3_prepare_scene_dataset.py`: Selección de escenas
- `scripts/4_clip_compute_embeddings.py`: Rutas de metadata, root_dir, output
- `scripts/5_train_vit_trainer.py`: Modo entrenamiento, escena, CLIP

**Formato de menús:**
```
Seleccione opción:
  [1] Opción A
  [2] Opción B
Ingrese opción (1/2, default 1): 
```

---

## 🗂️ ESTRUCTURA DE ARCHIVOS NUMERADOS

Todos los scripts están numerados del **0 al 7** para indicar orden de ejecución:

```
0. scripts/0_map_coords_to_continent.py      # Mapeo GPS → continente
1. scripts/1_prepare_dataset_folders.py      # Organización por continente
2. scripts/2_run_scene_filter.py             # Clasificación indoor/outdoor
3. scripts/3_prepare_scene_dataset.py        # Reorganización + splits
4. scripts/4_clip_compute_embeddings.py      # Embeddings CLIP (opcional)
5. scripts/5_train_vit_trainer.py            # Entrenamiento ⭐
6. scripts/6_eval_metrics.py                 # Evaluación
7. scripts/7_predict_image.py                # Predicción
```

**Archivos eliminados/no usados:**
- ❌ `scripts/train_clip_classifier.py` (eliminado, Opción B)
- ❌ `scripts/1_map_coords_to_continent.py` (duplicado, usar 0)
- ❌ `scripts/5_predict_image.py` (duplicado, usar 7)

---

## 📈 RESULTADOS ACTUALES

### Métricas Globales

**Con CLIP Fusionado (Opción A):**
- **Accuracy**: 75.21%
- **F1-Score (weighted)**: 74.88%
- **Precision (weighted)**: 74.85%
- **Recall (weighted)**: 75.21%
- **F1-Score (macro)**: 69.79%

**Sin CLIP (solo ViT + Geolocalización):**
- **Accuracy**: ~70.76%
- **F1-Score (weighted)**: ~69.71%

**Mejora con CLIP:** ~+5 puntos porcentuales en todas las métricas

### Performance por Continente (Outdoor, con CLIP)

| Continente | Precision | Recall | F1-Score | Support | Observaciones |
|-----------|-----------|--------|----------|---------|---------------|
| **África** | 0.6283 | 0.5441 | 0.5832 | 261 | Clase minoritaria, necesita más datos |
| **América** | 0.7577 | 0.7991 | 0.7779 | 1,354 | Mejor balance precision/recall |
| **Asia** | 0.7168 | 0.6048 | 0.6560 | 749 | Buena precisión, recall moderado |
| **Europa** | 0.8054 | 0.8568 | 0.8303 | 1,522 | Mejor rendimiento general |
| **Oceanía** | 0.6557 | 0.6295 | 0.6423 | 475 | Rendimiento estable |

**Análisis:**
- **Europa** y **América** tienen mejor rendimiento (mayor cantidad de datos)
- **África** y **Oceanía** tienen menor rendimiento (escasez de datos)
- El modelo generaliza bien en clases mayoritarias

---

## 🔍 PROBLEMAS RESUELTOS Y SOLUCIONES

### 1. Error: `ModuleNotFoundError: No module named 'preprocessing'`

**Causa:** Python no encontraba el paquete `preprocessing`.

**Solución:**
- Creado `preprocessing/__init__.py` (hace que sea un paquete)
- Agregado `ROOT_DIR` y `sys.path.insert()` en todos los scripts

**Patrón usado:**
```python
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
```

---

### 2. Error: `TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'`

**Causa:** Versión de `transformers` usa `eval_strategy` en lugar de `evaluation_strategy`.

**Solución:**
- Cambiado `evaluation_strategy` → `eval_strategy` en `scripts/5_train_vit_trainer.py`

---

### 3. Error: `FileNotFoundError: pytorch_model.bin` o `model.safetensors`

**Causa:** Script de evaluación buscaba pesos en ubicación incorrecta.

**Solución:**
- Guardar `checkpoint_dir` en `training_metadata.json` durante entrenamiento
- En evaluación, leer `checkpoint_dir` desde metadata
- Buscar pesos en `checkpoint_dir/model.safetensors` o `checkpoint_dir/pytorch_model.bin`
- Si no se encuentra, buscar en `checkpoint-*/` más reciente

---

### 4. Problema: Rendimiento bajo (~52%) con indoor/outdoor combinados

**Causa:** Imágenes indoor tienen mucho ruido para clasificación geográfica.

**Solución:**
- Separar entrenamiento: solo outdoor por defecto
- Resultado: F1-score sube a ~70% sin CLIP, ~75% con CLIP

---

### 5. Problema: Data leakage en splits

**Causa:** Splits se regeneraban en cada ejecución.

**Solución:**
- Splits persistentes en `data/train_<scene>/` y `data/test_<scene>/`
- Script 3 los genera una vez
- Script 5 lee desde estos splits directamente

---

## 🔑 PUNTOS CRÍTICOS DE CONFIGURACIÓN

### 1. Modelo Places365

**Ubicación requerida:**
```
models/places365/resnet50_places365.pth
```

**Descarga manual:**
- URL: http://places2.csail.mit.edu/models_places365/resnet50_places365.pth
- Tamaño: ~100 MB

**Si falta:** Script 2 lanza `FileNotFoundError` con mensaje claro.

---

### 2. Embeddings CLIP

**Formato:** Archivos `.npz` con:
- `embeddings`: numpy array (N, 512)
- `paths`: array de strings (rutas relativas)
- `model_name`: string (nombre del modelo CLIP)

**Ubicación típica:**
- `data/clip_embeddings/train_outdoor_embeddings.npz`
- `data/clip_embeddings/test_outdoor_embeddings.npz`

**Si faltan:** Script 5 pregunta si quieres integrar CLIP, si dices "y" pero no existen los archivos, deberás generarlos primero con script 4.

---

### 3. Metadata de Entrenamiento

**Formato:** JSON con configuración completa del modelo.

**Ubicación:** `outputs/checkpoints/vit-continent-balanced-<tag>/training_metadata.json`

**Contiene:**
- Rutas a datasets y metadatos
- Configuración de CLIP (si está habilitado)
- Configuración de geolocalización
- Path al mejor checkpoint

**Uso:** Scripts 6 y 7 leen este archivo para cargar configuración automáticamente.

---

### 4. Checkpoints

**Ubicación:** `outputs/checkpoints/vit-continent-balanced-<tag>/checkpoint-XXXX/`

**Archivos:**
- `model.safetensors` (formato safetensors, preferido)
- O `pytorch_model.bin` (formato PyTorch legacy)

**Mejor checkpoint:** Guardado en `training_metadata.json` como `checkpoint_dir`.

---

## 🧪 COMANDOS RÁPIDOS DE REFERENCIA

### Pipeline Completo (Ejecutar en orden)

```bash
# 1. Extraer imágenes
python downloader_smart.py

# 2. Mapear continentes
python scripts/0_map_coords_to_continent.py

# 3. Organizar por continente
python scripts/1_prepare_dataset_folders.py

# 4. Clasificar escenas
python scripts/2_run_scene_filter.py

# 5. Preparar splits (seleccionar [1] para outdoor)
python scripts/3_prepare_scene_dataset.py

# 6. Generar embeddings CLIP - Train
python scripts/4_clip_compute_embeddings.py
# Proporcionar: data/metadata/train_outdoor_metadata.csv
#              data/train_outdoor
#              data/clip_embeddings/train_outdoor_embeddings.npz

# 7. Generar embeddings CLIP - Test
python scripts/4_clip_compute_embeddings.py
# Proporcionar: data/metadata/test_outdoor_metadata.csv
#              data/test_outdoor
#              data/clip_embeddings/test_outdoor_embeddings.npz

# 8. Entrenar (seleccionar [1], [1], y)
python scripts/5_train_vit_trainer.py

# 9. Evaluar
python scripts/6_eval_metrics.py

# 10. Predecir
python scripts/7_predict_image.py --image ruta/a/imagen.jpg --lat 40.7128 --lon -74.0060
```

### Comandos Útiles

```bash
# Forzar recálculo completo
python scripts/0_map_coords_to_continent.py --full-refresh
python scripts/1_prepare_dataset_folders.py --full-refresh

# Evaluar modelo específico
python scripts/6_eval_metrics.py --model-dir outputs/checkpoints/vit-continent-balanced-outdoor

# Predecir con coordenadas
python scripts/7_predict_image.py --image foto.jpg --lat 40.7128 --lon -74.0060 --probs
```

---

## 📦 DEPENDENCIAS Y REQUISITOS

### requirements.txt

```
torch
torchvision
transformers
datasets
pandas
numpy
tqdm
matplotlib
scikit-learn
geopandas
cartopy
shapely
albumentations
pillow
```

### Instalación PyTorch con CUDA

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Modelos Pre-entrenados Necesarios

1. **ViT**: Descargado automáticamente por Hugging Face (`google/vit-base-patch16-224-in21k`)
2. **CLIP**: Descargado automáticamente por Hugging Face (`openai/clip-vit-base-patch32`)
3. **Places365**: Descarga manual requerida (ver sección "Puntos Críticos")

---

## 🎯 DECISIONES DE DISEÑO IMPORTANTES

### 1. Numeración de Scripts

**Decisión:** Scripts numerados del 0 al 7 para indicar orden de ejecución.

**Razón:** Facilita seguimiento del pipeline y ejecución ordenada.

**Nota:** Script 4 (CLIP embeddings) es opcional pero está numerado para mantener secuencia.

---

### 2. Separación de Preprocesamiento

**Decisión:** Módulos de preprocesamiento en carpeta `preprocessing/`.

**Contenido:**
- `scene_classifier.py`: Clasificador Places365
- `geo_dataset.py`: Dataset PyTorch con geo features y CLIP

**Ventaja:** Código reutilizable, fácil de importar.

---

### 3. Metadata Persistente

**Decisión:** Guardar configuración completa en `training_metadata.json`.

**Ventaja:**
- Scripts 6 y 7 pueden cargar configuración automáticamente
- Reproducibilidad: saber exactamente cómo se entrenó el modelo
- Facilita deployment: toda la información está disponible

---

### 4. Formato de Checkpoints

**Decisión:** Usar `safetensors` como formato preferido.

**Razón:** Más seguro y eficiente que `pytorch_model.bin`.

**Fallback:** Script de evaluación busca ambos formatos.

---

## 🚨 CASOS ESPECIALES Y NOTAS

### 1. Coordenadas Faltantes (NaN)

**Manejo:**
- `compute_lat_bin()` y `compute_lon_bin()` retornan `UNKNOWN_BIN = -1` si lat/lon es NaN
- En dataset, `UNKNOWN_BIN` se mapea al último bin (LAT_BINS_TOTAL - 1, LON_BINS_TOTAL - 1)
- El modelo aprende a manejar coordenadas faltantes

---

### 2. Imágenes Sin Embedding CLIP

**En entrenamiento:**
- Si una imagen no tiene embedding CLIP, se usa tensor de ceros (512 dim)
- El modelo maneja esto correctamente

**En evaluación:**
- Si falta embedding CLIP en metadata pero modelo fue entrenado con CLIP, se genera warning

---

### 3. Normalización de Continentes

**Mapeo:**
- "North America" → "Americas"
- "South America" → "Americas"
- Otros continentes: Sin cambios

**Implementado en:**
- Script 0 (mapeo geográfico)
- Script 1 (organización)
- Script 2 (scene filter)

---

### 4. Formato de Nombres de Archivos

**Convención:**
- Imágenes extraídas: `shard0_00000.jpg`, `shard0_00001.jpg`, etc.
- Índice: 5 dígitos con padding ceros
- **Importante**: Todos los shards usan prefijo `shard0_` para compatibilidad

---

## 🔐 VARIABLES DE ENTORNO (Opcionales)

Algunos scripts pueden usar variables de entorno como fallback:

- `TRAINING_MODE`: "scene" o "scene_continent"
- `SCENE_TYPE`: "outdoor" o "indoor"

**Uso:** Solo si no se pasan argumentos ni se usan menús interactivos.

---

## 📝 NOTAS DE MANTENIMIENTO

### Archivos a NO Modificar Manualmente

1. **Splits train/test**: Generados por script 3, no modificar manualmente
2. **training_metadata.json**: Generado por script 5, puede leerse pero no modificarse
3. **CSVs de metadata**: Generados por script 3, contienen bins calculados

### Archivos Seguros para Modificar

1. **downloader_smart.py**: Configuración de shards y filtros
2. **Hiperparámetros en script 5**: Pueden ajustarse según necesidad

---

## 🎓 CONTEXTO DEL DESARROLLO

### Orden de Implementación (Cronológico)

1. **Pipeline básico**: ViT simple sin geo features ni CLIP
2. **Geolocalización**: Agregado lat/lon bins como features
3. **Separación indoor/outdoor**: Implementado Places365 para filtrar
4. **CLIP Ensemble (Opción B)**: Implementado y probado (~75% F1)
5. **CLIP Fusionado (Opción A)**: Implementado, mejor que ensemble
6. **Eliminación Opción B**: Simplificación, solo queda Opción A
7. **Auto-incremental**: Agregado a scripts de procesamiento
8. **Menús interactivos**: Agregados para facilitar uso
9. **Numeración scripts**: Reorganizados del 0 al 7

### Resultados por Iteración

- **Iteración 1 (ViT básico)**: ~70% F1-score
- **Iteración 2 (Geo features)**: ~70% F1-score (sin mejora significativa)
- **Iteración 3 (Indoor/outdoor separados)**: ~52% F1-score (empeoró)
- **Iteración 4 (Solo outdoor)**: ~70% F1-score (mejoró)
- **Iteración 5 (CLIP Ensemble)**: ~75% F1-score
- **Iteración 6 (CLIP Fusionado)**: ~75% F1-score (mejor que ensemble)

**Lección aprendida:** Separar indoor/outdoor como clases de salida empeora rendimiento. Usar solo outdoor para entrenar mejora significativamente.

---

## 🔄 ESTADO ACTUAL DEL PROYECTO

### ✅ Completado

- [x] Pipeline completo 0-7 funcional
- [x] CLIP fusionado implementado
- [x] Geolocalización integrada
- [x] Clasificación indoor/outdoor
- [x] Auto-incremental en procesamiento
- [x] Menús interactivos
- [x] Splits persistentes
- [x] Early stopping
- [x] Balanceo de clases
- [x] Documentación completa (README + GUIA_EJECUCION.md)

### 📊 Resultados Actuales

- **F1-Score con CLIP**: 74.88% (weighted)
- **F1-Score sin CLIP**: 69.71% (weighted)
- **Mejora con CLIP**: +5.17 puntos porcentuales

### 🔮 Posibles Mejoras Futuras

- [ ] Más datos para África y Oceanía
- [ ] Experimentar con ViT-Large o Swin Transformer
- [ ] Fine-tuning de CLIP (no solo proyección)
- [ ] Ajuste de hiperparámetros (especialmente clip_projection_dim)
- [ ] OCR para texto en imágenes
- [ ] Features temporales (hora del día)

---

## 💡 TRUCOS Y MEJORES PRÁCTICAS

### 1. Ejecutar Scripts sin Argumentos

Todos los scripts tienen valores por defecto y/o menús interactivos, así que puedes ejecutarlos directamente:
```bash
python scripts/5_train_vit_trainer.py
```

El script te guiará con menús interactivos.

---

### 2. Verificar Estado del Pipeline

Después de cada script, verifica:
- **Script 0**: Revisa `coords_with_continent.csv` y distribución por continente
- **Script 1**: Revisa `data/images_by_continent/<continent>/` tiene imágenes
- **Script 2**: Revisa `outputs/scene_predictions.csv` y distribución indoor/outdoor
- **Script 3**: Revisa `data/metadata/*.csv` tienen columnas `lat_bin` y `lon_bin`
- **Script 4**: Revisa archivo `.npz` generado tiene embeddings y paths
- **Script 5**: Revisa `training_metadata.json` tiene toda la configuración
- **Script 6**: Revisa `outputs/classification_report.txt` y `confusion_matrix.png`
- **Script 7**: Prueba con varias imágenes

---

### 3. Re-entrenar con Nuevos Datos

**Proceso:**
1. Ejecutar `downloader_smart.py` (agrega nuevas imágenes)
2. Ejecutar script 0 con `--full-refresh` O sin flag (incremental)
3. Ejecutar script 1 con `--full-refresh` O sin flag (incremental)
4. Ejecutar scripts 2-3 (reprocesar desde escenas)
5. Opcional: Re-generar embeddings CLIP (script 4)
6. Re-entrenar (script 5)

---

### 4. Verificar GPU

```python
import torch
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

**GPU recomendada:** RTX 3050 o superior para entrenamiento razonable.

---

### 5. Reducir Memoria GPU

Si tienes errores de memoria:
- Reducir `BATCH_SIZE` en `scripts/5_train_vit_trainer.py` (línea 152): de 16 a 8 o 4
- Reducir `batch_size` en `scripts/2_run_scene_filter.py` (línea 108): de 32 a 16
- Reducir `batch_size` en `scripts/4_clip_compute_embeddings.py` (línea 47): de 64 a 32

---

## 🐛 DEBUGGING COMÚN

### Problema: "No images found to classify"

**Causa:** `data/images_by_continent/` está vacío o no tiene subcarpetas.

**Solución:** Ejecutar script 1 primero para organizar imágenes.

---

### Problema: "FileNotFoundError: training_metadata.json"

**Causa:** Modelo no fue entrenado o metadata no existe.

**Solución:** Ejecutar script 5 para entrenar modelo primero.

---

### Problema: "Dimensiones distintas entre embeddings CLIP de train y test"

**Causa:** Embeddings de train y test fueron generados con diferentes modelos CLIP o tienen diferente dimensión.

**Solución:** Re-generar embeddings con mismo modelo CLIP para train y test.

---

### Problema: Imágenes faltantes en dataset

**Causa:** Algunas imágenes fueron eliminadas o movidas después de generar metadata.

**Solución:** Re-ejecutar script 3 para regenerar metadata y splits.

---

## 📚 REFERENCIAS Y RECURSOS

### Modelos Pre-entrenados

- **ViT**: https://huggingface.co/google/vit-base-patch16-224-in21k
- **CLIP**: https://huggingface.co/openai/clip-vit-base-patch32
- **Places365**: http://places2.csail.mit.edu/models_places365.html

### Datasets

- **Large Dataset of Geotagged Images**: https://www.kaggle.com/datasets/habedi/large-dataset-of-geotagged-images

### Bibliotecas Principales

- **Hugging Face Transformers**: https://github.com/huggingface/transformers
- **Albumentations**: https://github.com/albumentations-team/albumentations
- **GeoPandas**: https://geopandas.org/

---

## 📞 INFORMACIÓN DE CONTACTO Y AUTORES

**Autores:**
- **Agustín Figueroa**
- **Esteban Alvarez**

**Video del Proyecto:**
https://youtu.be/8sCkD-1eMto?si=e50IRY_O9UGkO97z

---

## 🔄 RESUMEN PARA NUEVO CHAT

Si estás iniciando un nuevo chat, proporciona esta información:

1. **Este archivo completo** (`CONTEXTO_PROYECTO.md`)
2. **Estado actual:** Pipeline completo funcional, CLIP fusionado implementado, F1-score ~75%
3. **Archivos críticos recientes:**
   - `scripts/5_train_vit_trainer.py` (entrenamiento principal)
   - `scripts/6_eval_metrics.py` (evaluación)
   - `scripts/7_predict_image.py` (predicción)
   - `preprocessing/geo_dataset.py` (dataset con geo features y CLIP)
4. **Decisiones importantes:**
   - Solo Opción A (Fusión CLIP) implementada
   - Entrenar solo con outdoor (mejor rendimiento)
   - Splits persistentes (no regenerar)
   - Auto-incremental en scripts 0 y 1

---

**Última actualización:** Después de recuperación de archivos faltantes (scripts 2, 4, 5)
**Estado:** Todos los scripts funcionando correctamente con todas las funcionalidades implementadas

