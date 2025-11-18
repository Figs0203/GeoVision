# GeoVision: Clasificación Geográfica con Vision Transformer

Proyecto de clasificación de imágenes por continente utilizando Vision Transformer (ViT) con integración de CLIP y geolocalización. El modelo identifica el continente de origen de una imagen con una precisión del ~75% usando CLIP fusionado.

## Video del Proyecto

https://youtu.be/8sCkD-1eMto?si=e50IRY_O9UGkO97z

## Descripción del Proyecto

GeoVision utiliza un modelo Vision Transformer (ViT) pre-entrenado y fine-tuneado con **CLIP fusionado** y **features de geolocalización** para clasificar imágenes geográficas en 5 continentes: **África, América, Asia, Europa y Oceanía**. El proyecto incluye todo el pipeline desde la extracción de datos hasta la evaluación y predicción, con un sistema de clasificación previa indoor/outdoor para mejorar la calidad del dataset.

### Características Principales

- **Modelo Principal**: Vision Transformer (ViT-base-patch16-224) pre-entrenado en ImageNet-21k
- **CLIP Fusionado (Opción A)**: Integración de embeddings CLIP directamente en el modelo ViT durante el entrenamiento
- **Geolocalización**: Features de latitud/longitud discretizadas (bins) como contexto geográfico
- **Clasificación de Escenas**: Pre-filtrado indoor/outdoor usando Places365 ResNet-50 para mejorar calidad del dataset
- **Dataset**: 37,000+ imágenes geoetiquetadas del [Large Dataset of Geotagged Images](https://www.kaggle.com/datasets/habedi/large-dataset-of-geotagged-images)
- **Precisión**: 
  - ~75% F1-score con CLIP fusionado
  - ~70% F1-score sin CLIP (solo ViT + geolocalización)
- **Técnicas Avanzadas**: 
  - Data augmentation con Albumentations
  - Class balancing con WeightedRandomSampler
  - Mixed precision training (FP16)
  - Early stopping para prevenir overfitting
  - Scheduler cosine para optimización de learning rate
  - GPU acceleration

## Estructura del Proyecto

```
GeoVision/
├── downloader_smart.py                   # Extracción inteligente de imágenes del shard
├── scripts/
│   ├── 0_map_coords_to_continent.py      # Mapeo de coordenadas a continentes
│   ├── 1_prepare_dataset_folders.py      # Organización inicial por continente
│   ├── 2_run_scene_filter.py             # Clasificación indoor/outdoor (Places365)
│   ├── 3_prepare_scene_dataset.py        # Reorganización por escena + continente
│   ├── 4_clip_compute_embeddings.py      # Generación de embeddings CLIP (opcional)
│   ├── 5_train_vit_trainer.py            # Entrenamiento del modelo ViT con CLIP
│   ├── 6_eval_metrics.py                 # Evaluación y métricas
│   └── 7_predict_image.py                # Predicción en imágenes nuevas
├── preprocessing/
│   ├── scene_classifier.py               # Clasificador Places365 (indoor/outdoor)
│   └── geo_dataset.py                    # Dataset con features geográficas y CLIP
├── models/
│   └── places365/                        # Modelo Places365 (descargar manualmente)
│       ├── resnet50_places365.pth
│       ├── categories_places365.txt
│       └── IO_places365.txt
├── shards/                               # Shards de MessagePack (descargar de Kaggle)
├── images/                               # Imágenes extraídas (generado)
├── data/
│   ├── images_by_continent/              # Dataset organizado por continente
│   ├── images_by_scene/                  # Dataset organizado por escena/continente
│   ├── train_outdoor/                    # Split de entrenamiento (persistente)
│   ├── test_outdoor/                     # Split de prueba (persistente)
│   ├── metadata/                         # CSVs con metadatos y bins geográficos
│   └── clip_embeddings/                  # Embeddings CLIP pre-computados (.npz)
├── outputs/
│   ├── checkpoints/                      # Modelos entrenados (generado)
│   ├── confusion_matrix.png              # Matriz de confusión
│   ├── classification_report.txt         # Reporte detallado
│   └── scene_predictions.csv             # Clasificaciones indoor/outdoor
├── coords.csv                            # Coordenadas extraídas (generado)
├── coords_with_continent.csv             # Coordenadas + continentes (generado)
├── requirements.txt                      # Dependencias
├── GUIA_EJECUCION.md                     # Guía detallada paso a paso
└── README.md
```

## Instalación

### Requisitos Previos

- Python 3.8+
- CUDA 11.8+ (opcional, recomendado para GPU)
- 16GB+ RAM recomendado
- ~15GB espacio en disco para dataset, modelos y embeddings
- GPU NVIDIA (RTX 3050 o superior recomendado)

### Pasos de Instalación

1. **Clonar el repositorio**

```bash
git clone https://github.com/Figs0203/GeoVision.git
cd GeoVision
```

2. **Crear entorno virtual**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Instalar dependencias**

```bash
pip install -r requirements.txt
```

4. **Instalar PyTorch con CUDA (recomendado, para GPU)**

```bash
# Para CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Para CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Solo CPU (no recomendado para entrenamiento)
pip install torch torchvision
```

5. **Descargar Modelo Places365**

Para la clasificación indoor/outdoor, descarga el modelo Places365:

- **Modelo**: [resnet50_places365.pth](http://places2.csail.mit.edu/models_places365/resnet50_places365.pth)
- **Categorías**: `categories_places365.txt` (incluido en el repositorio)
- **Split IO**: `IO_places365.txt` (incluido en el repositorio)

Coloca el archivo `resnet50_places365.pth` en `models/places365/`

## Guía Rápida de Ejecución

Para una guía completa y detallada paso a paso, consulta **[GUIA_EJECUCION.md](GUIA_EJECUCION.md)**.

### Pipeline Completo (Resumen)

```bash
# 1. Extraer imágenes del shard
python downloader_smart.py

# 2. Mapear coordenadas a continentes
python scripts/0_map_coords_to_continent.py

# 3. Organizar en carpetas por continente
python scripts/1_prepare_dataset_folders.py

# 4. Clasificar imágenes como indoor/outdoor
python scripts/2_run_scene_filter.py

# 5. Reorganizar por escena + continente y crear splits
python scripts/3_prepare_scene_dataset.py
# Selecciona [1] para solo outdoor (recomendado)

# 6. (Opcional) Generar embeddings CLIP para train
python scripts/4_clip_compute_embeddings.py
# Proporciona: data/metadata/train_outdoor_metadata.csv

# 7. (Opcional) Generar embeddings CLIP para test
python scripts/4_clip_compute_embeddings.py
# Proporciona: data/metadata/test_outdoor_metadata.csv

# 8. Entrenar modelo ViT (con o sin CLIP)
python scripts/5_train_vit_trainer.py
# Selecciona [1] para outdoor, [y] para integrar CLIP si completaste pasos 6-7

# 9. Evaluar modelo
python scripts/6_eval_metrics.py

# 10. Predecir en nueva imagen
python scripts/7_predict_image.py --image ruta/a/imagen.jpg
```

## Descripción de Componentes

### 1. Extracción de Datos (`downloader_smart.py`)

Extrae imágenes desde archivos `.msg` (MessagePack) y genera CSV con coordenadas GPS.

**Características:**
- Modo incremental: continúa desde último índice procesado
- Filtrado por continente (opcional)
- Genera `coords.csv` con filename, lat, lon

### 2. Mapeo Geográfico (`scripts/0_map_coords_to_continent.py`)

Mapea coordenadas GPS a nombres de continentes usando `geopandas` y `cartopy`.

**Características:**
- Modo incremental: solo procesa nuevas coordenadas
- Genera `coords_with_continent.csv`
- Soporta `--full-refresh` para recálculo completo

### 3. Organización de Dataset (`scripts/1_prepare_dataset_folders.py`)

Organiza imágenes en estructura `data/images_by_continent/<continent>/`.

**Características:**
- Modo incremental: solo copia imágenes nuevas
- Opción para generar splits train/test
- Soporta `--full-refresh`

### 4. Clasificación de Escenas (`scripts/2_run_scene_filter.py`)

Clasifica cada imagen como `indoor`, `outdoor` o `unknown` usando Places365 ResNet-50.

**Características:**
- **Filtrado por confianza mínima**: Si `confidence < 0.6` (configurable), marca como `unknown`
- Filtrado automático de imágenes `unknown` (incluye las de baja confianza)
- Genera `outputs/scene_predictions.csv` con scene_type, confidence, continent
- Acelera el proceso usando GPU
- **Mejora calidad del dataset**: Elimina imágenes ambiguas (ej: interiores con ventanas grandes) que Places365 clasifica incorrectamente

### 5. Preparación de Splits (`scripts/3_prepare_scene_dataset.py`)

Reorganiza dataset en estructura jerárquica `scene/continent/` y genera splits persistentes.

**Características:**
- Genera splits train/test persistentes (evita data leakage)
- Crea metadatos con `lat_bin` y `lon_bin` para features geográficas
- Menú interactivo para seleccionar escenas (indoor/outdoor/ambos)
- **Recomendación**: Usar solo `outdoor` (imágenes indoor tienen mucho ruido)

### 6. Embeddings CLIP (`scripts/4_clip_compute_embeddings.py`) - OPCIONAL

Pre-computa embeddings CLIP para acelerar entrenamiento y mejorar precisión.

**Características:**
- Genera embeddings de 512 dimensiones usando `openai/clip-vit-base-patch32`
- Guarda en formato `.npz` para uso posterior
- Menú interactivo para configurar rutas
- **Recomendado**: Generar para train y test antes de entrenar

### 7. Entrenamiento (`scripts/5_train_vit_trainer.py`)

Entrena el modelo ViT con geolocalización y opcionalmente CLIP fusionado.

**Características:**
- Modelo: `GeoViTForImageClassification` con:
  - Backbone ViT (768 dim)
  - Embeddings geográficos (lat/lon bins, 32 dim cada uno)
  - Proyección CLIP (128 dim, si está habilitado)
- Hiperparámetros:
  - Learning rate: 3e-5
  - Épocas: 12 (con early stopping, patience=5)
  - Batch size: 16
  - Weight decay: 0.01
  - Scheduler: cosine con warmup
- Guarda checkpoints cada 500 steps
- Early stopping basado en F1-score

### 8. Evaluación (`scripts/6_eval_metrics.py`)

Genera métricas de rendimiento completas.

**Características:**
- Lee configuración automáticamente desde `training_metadata.json`
- Genera matriz de confusión visualizada
- Reporte de clasificación detallado (precision, recall, F1-score)
- Colapsa etiquetas scene-continent automáticamente (opcional)

### 9. Predicción (`scripts/7_predict_image.py`)

Clasifica una imagen nueva por continente.

**Características:**
- Carga modelo más reciente automáticamente
- Soporta coordenadas geográficas opcionales (mejora precisión)
- Genera embeddings CLIP on-the-fly si el modelo fue entrenado con CLIP
- Muestra probabilidades para todas las clases

## Resultados

### Métricas Globales

**Con CLIP Fusionado (Opción A):**
- **Accuracy**: ~75.2%
- **F1-Score**: ~74.9% (weighted)
- **Precision**: ~74.9% (weighted)
- **Recall**: ~75.2% (weighted)

**Sin CLIP (solo ViT + Geolocalización):**
- **Accuracy**: ~70.8%
- **F1-Score**: ~69.7% (weighted)

### Performance por Continente (Outdoor, con CLIP)

| Continente | Precision | Recall | F1-Score | Support | Observaciones |
|-----------|-----------|--------|----------|---------|---------------|
| **África** | 0.6283 | 0.5441 | 0.5832 | 261 | Clase minoritaria, necesita más datos |
| **América** | 0.7577 | 0.7991 | 0.7779 | 1,354 | Mejor balance entre precision/recall |
| **Asia** | 0.7168 | 0.6048 | 0.6560 | 749 | Buena precisión, recall moderado |
| **Europa** | 0.8054 | 0.8568 | 0.8303 | 1,522 | Mejor rendimiento general |
| **Oceanía** | 0.6557 | 0.6295 | 0.6423 | 475 | Rendimiento estable |

**Análisis:**
- **Europa** y **América** tienen mejor rendimiento debido a mayor cantidad de datos de entrenamiento
- **África** y **Oceanía** tienen menor rendimiento por escasez de datos
- El modelo muestra buena capacidad de generalización en clases mayoritarias

### Comparación: Con vs Sin CLIP

| Métrica | Sin CLIP | Con CLIP (Fusión) | Mejora |
|---------|----------|-------------------|--------|
| Accuracy | 70.76% | 75.21% | +4.45% |
| F1-Score | 69.71% | 74.88% | +5.17% |
| Precision | 69.91% | 74.85% | +4.94% |
| Recall | 70.76% | 75.21% | +4.45% |

**Conclusión**: La integración de CLIP fusionado proporciona una mejora consistente de ~5 puntos porcentuales en todas las métricas.

## Arquitectura del Modelo

El modelo `GeoViTForImageClassification` combina múltiples fuentes de información:

```
Imagen (224x224) → ViT Backbone → [768 dim]
                                           ↓
Lat/Lon → Embeddings Geo → [32 + 32 = 64 dim]
                                           ↓
CLIP Embedding → Proyección → [128 dim]
                                           ↓
        ┌────────────────────────────┐
        │ Concatenación + Dropout    │ → [768 + 64 + 128 = 960 dim]
        └────────────────────────────┘
                       ↓
            Classifier (5 clases)
```

**Componentes:**
- **ViT Backbone**: Características visuales de la imagen
- **Geolocalización**: Bins de latitud/longitud (10° por bin)
- **CLIP (opcional)**: Embeddings semánticos proyectados a 128 dim

## Dataset

**Fuente:** [Large Dataset of Geotagged Images](https://www.kaggle.com/datasets/habedi/large-dataset-of-geotagged-images)

**Características:**
- Formato: MessagePack (.msg) dividido en shards
- Shards utilizados: shard_0, shard_2, shard_3 (37,000+ imágenes)
- Metadatos: Latitud, Longitud por cada imagen
- Distribución: Desbalanceada (Europa y América dominan ~84%)
- Filtrado: Solo imágenes outdoor (mejora calidad del dataset)

**Procesamiento:**
1. Extracción de imágenes JPEG desde MessagePack
2. Mapeo de coordenadas GPS a continentes
3. Clasificación indoor/outdoor (Places365)
4. Filtrado de imágenes con baja confianza (< 0.6) y `unknown`
5. Filtrado de imágenes `indoor` (opcional, recomendado: solo outdoor)
6. División estratificada 80/20 train/test (persistente)

## Tecnologías Utilizadas

### Core
- **PyTorch**: Framework de deep learning
- **Transformers (Hugging Face)**: Implementación de Vision Transformer y CLIP
- **Datasets (Hugging Face)**: Gestión eficiente de datasets

### Modelos Pre-entrenados
- **ViT**: [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k)
- **CLIP**: [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)
- **Places365**: ResNet-50 pre-entrenado en Places365

### Procesamiento de Datos
- **MessagePack**: Deserialización del shard
- **Pillow (PIL)**: Manipulación de imágenes
- **Pandas**: Manejo de datos tabulares
- **Albumentations**: Data augmentation avanzado

### Geolocalización
- **geopandas**: Procesamiento de datos geoespaciales
- **cartopy**: Acceso a datos de formas geográficas
- **shapely**: Operaciones geométricas

### Evaluación y Visualización
- **scikit-learn**: Métricas de clasificación
- **matplotlib/seaborn**: Visualización de resultados
- **tqdm**: Barras de progreso

## Mejoras Futuras

- [ ] Incorporar más shards para aumentar dataset (100K+ imágenes)
- [ ] Experimentar con modelos más grandes (ViT-Large, Swin Transformer)
- [ ] Fine-tuning de CLIP en lugar de solo proyección
- [ ] Agregar clasificación jerárquica (continente → país → ciudad)
- [ ] Implementar explicabilidad con Grad-CAM/Attention maps
- [ ] Desplegar como API REST con FastAPI
- [ ] Crear interfaz web con Streamlit/Gradio
- [ ] Balancear dataset con técnicas de oversampling/undersampling avanzadas
- [ ] Incorporar features adicionales (OCR para texto en imágenes, hora del día, clima)
- [ ] Experimentar con diferentes dimensiones de proyección CLIP

## Autores

**Agustín Figueroa**  
**Esteban Alvarez**

## Agradecimientos

- **Dataset**: [habedi/large-dataset-of-geotagged-images](https://www.kaggle.com/datasets/habedi/large-dataset-of-geotagged-images) en Kaggle
- **Modelo ViT**: [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k)
- **Modelo CLIP**: [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)
- **Modelo Places365**: MIT Places Dataset Team
- **Hugging Face**: Por la biblioteca Transformers y Datasets
- **PyTorch Team**: Por el framework de deep learning

