"""
Fine-tune ViT using Hugging Face Trainer on folder-structured images,
with Albumentations augmentations and real class balancing via WeightedRandomSampler.
Supports CLIP fusion and geolocation features.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import Counter
from typing import List, Optional

import numpy as np
import pandas as pd
from transformers import (
    AutoImageProcessor,
    TrainingArguments,
    Trainer,
    ViTModel,
    EarlyStoppingCallback,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import accuracy_score, f1_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from preprocessing.geo_dataset import (
    GeoImageDataset,
    LAT_BIN_SIZE,
    LON_BIN_SIZE,
    NUM_LAT_BINS,
    NUM_LON_BINS,
    UNKNOWN_BIN,
    load_clip_embeddings,
)


# =====================
# CONFIGURACIÓN GENERAL
# =====================

def ask_yes_no(message: str, default: bool = False) -> bool:
    default_str = "Y/n" if default else "y/N"
    response = input(f"{message} ({default_str}): ").strip().lower()
    if not response:
        return default
    return response in {"y", "yes", "s", "si"}


def prompt_path(message: str, default: Path | None = None) -> Path:
    prompt_msg = message
    if default is not None:
        prompt_msg += f" [{default}]"
    prompt_msg += ": "
    value = input(prompt_msg).strip()
    if not value:
        if default is None:
            raise ValueError("Debe proporcionar un valor.")
        value = str(default)
    path = Path(value)
    # Resolver ruta relativa contra ROOT_DIR si no es absoluta
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path.resolve()


def _resolve_training_config() -> tuple:
    parser = argparse.ArgumentParser(
        description="Entrenar ViT para clasificación por continente."
    )
    parser.add_argument(
        "--training-mode",
        choices=["scene", "scene_continent"],
        help="Modo de entrenamiento. Si no se especifica, se solicitará por consola.",
    )
    parser.add_argument(
        "--scene-type",
        choices=["outdoor", "indoor"],
        help="Escena a usar cuando el modo es 'scene'. Si no se especifica, se solicitará.",
    )
    parser.add_argument(
        "--clip-train-embeddings",
        type=Path,
        help="Archivo .npz con embeddings CLIP para entrenamiento (opcional).",
    )
    parser.add_argument(
        "--clip-test-embeddings",
        type=Path,
        help="Archivo .npz con embeddings CLIP para validación (opcional).",
    )
    parser.add_argument(
        "--clip-projection-dim",
        type=int,
        default=128,
        help="Dimensión interna usada para proyectar embeddings CLIP.",
    )
    parser.add_argument(
        "--clip-model-name",
        type=str,
        help="Identificador del modelo CLIP usado para generar embeddings.",
    )
    args = parser.parse_args()

    training_mode = (args.training_mode or os.getenv("TRAINING_MODE") or "").lower()
    if training_mode not in {"scene", "scene_continent"}:
        print("Seleccione modo de entrenamiento:")
        print("  [1] Solo exteriores/interiores (scene)")
        print("  [2] Escena + continente combinado (scene_continent)")
        option = input("Ingrese opción (1/2, default 1): ").strip() or "1"
        training_mode = "scene" if option == "1" else "scene_continent"

    scene_default = (args.scene_type or os.getenv("SCENE_TYPE") or "outdoor").lower()
    if training_mode == "scene":
        print("Seleccione escena a utilizar:")
        print("  [1] Outdoor")
        print("  [2] Indoor")
        prompt = f"Ingrese opción (1/2, default {'1' if scene_default == 'outdoor' else '2'}): "
        option = input(prompt).strip()
        if not option:
            option = "1" if scene_default == "outdoor" else "2"
        if option not in {"1", "2"}:
            raise ValueError("Opción inválida. Debe ser 1 o 2.")
        scene_type = "outdoor" if option == "1" else "indoor"
    else:
        scene_type = scene_default

    return args, training_mode, scene_type


ARGS, TRAINING_MODE, SCENE_TYPE = _resolve_training_config()

if TRAINING_MODE == "scene":
    TRAIN_DIR = Path(f"data/train_{SCENE_TYPE}")
    TEST_DIR = Path(f"data/test_{SCENE_TYPE}")
    MODEL_TAG = SCENE_TYPE
elif TRAINING_MODE == "scene_continent":
    TRAIN_DIR = Path("data/train_scene_continent")
    TEST_DIR = Path("data/test_scene_continent")
    MODEL_TAG = "scene-continent"
else:
    raise ValueError(
        f"TRAINING_MODE desconocido '{TRAINING_MODE}'. Usa 'scene' o 'scene_continent'."
    )

MODEL_NAME = 'google/vit-base-patch16-224-in21k'
OUT_DIR = f'outputs/checkpoints/vit-continent-balanced-{MODEL_TAG}'
BATCH_SIZE = 16
NUM_EPOCHS = 12


# =====================
# 1) Cargar dataset
# =====================
print("\n[1/10] Cargando dataset desde carpetas persistentes...")
print(f"Modo de entrenamiento: {TRAINING_MODE}")
if not TRAIN_DIR.exists() or not TEST_DIR.exists():
    raise FileNotFoundError(
        f"No se encontraron los directorios de split persistente {TRAIN_DIR} y {TEST_DIR}. "
        "Ejecuta 3_prepare_scene_dataset.py con generación de splits."
    )

metadata_dir = Path("data/metadata")
if TRAINING_MODE == "scene":
    train_metadata_path = metadata_dir / f"train_{SCENE_TYPE}_metadata.csv"
    test_metadata_path = metadata_dir / f"test_{SCENE_TYPE}_metadata.csv"
else:
    train_metadata_path = metadata_dir / "train_scene_continent_metadata.csv"
    test_metadata_path = metadata_dir / "test_scene_continent_metadata.csv"

if not train_metadata_path.exists() or not test_metadata_path.exists():
    raise FileNotFoundError(
        f"No se encontraron metadatos de entrenamiento ({train_metadata_path}) o prueba ({test_metadata_path}). "
        "Ejecuta 3_prepare_scene_dataset.py para generarlos."
    )

train_meta_df = pd.read_csv(train_metadata_path)
test_meta_df = pd.read_csv(test_metadata_path)

use_clip = False
clip_train_embeddings_path = ARGS.clip_train_embeddings
clip_test_embeddings_path = ARGS.clip_test_embeddings
clip_projection_dim = ARGS.clip_projection_dim
clip_model_name = ARGS.clip_model_name
clip_input_dim = 0
clip_train_map = None
clip_test_map = None

if clip_train_embeddings_path and clip_test_embeddings_path:
    use_clip = True
else:
    if ask_yes_no("¿Deseas integrar embeddings CLIP durante el entrenamiento?", default=False):
        use_clip = True
        if clip_train_embeddings_path is None:
            clip_train_embeddings_path = prompt_path(
                "Archivo .npz con embeddings CLIP de entrenamiento",
                Path("data/clip_embeddings/train_outdoor_metadata_embeddings.npz"),
            )
        if clip_test_embeddings_path is None:
            clip_test_embeddings_path = prompt_path(
                "Archivo .npz con embeddings CLIP de validación",
                Path("data/clip_embeddings/test_outdoor_metadata_embeddings.npz"),
            )

if use_clip:
    # Resolver rutas si son relativas
    if clip_train_embeddings_path and not clip_train_embeddings_path.is_absolute():
        clip_train_embeddings_path = (ROOT_DIR / clip_train_embeddings_path).resolve()
    if clip_test_embeddings_path and not clip_test_embeddings_path.is_absolute():
        clip_test_embeddings_path = (ROOT_DIR / clip_test_embeddings_path).resolve()
    
    # Verificar que los archivos existan
    if not clip_train_embeddings_path.exists():
        raise FileNotFoundError(
            f"Archivo de embeddings CLIP de entrenamiento no encontrado: {clip_train_embeddings_path}\n"
            f"Por favor, ejecuta primero el script 4 (scripts/4_clip_compute_embeddings.py) para generar los embeddings."
        )
    if not clip_test_embeddings_path.exists():
        raise FileNotFoundError(
            f"Archivo de embeddings CLIP de validación no encontrado: {clip_test_embeddings_path}\n"
            f"Por favor, ejecuta primero el script 4 (scripts/4_clip_compute_embeddings.py) para generar los embeddings."
        )
    
    print(f"\nCargando embeddings CLIP...")
    print(f"  Entrenamiento: {clip_train_embeddings_path}")
    print(f"  Validación: {clip_test_embeddings_path}")
    
    clip_train_map, clip_input_dim, clip_model_name_train = load_clip_embeddings(clip_train_embeddings_path)
    clip_test_map, clip_test_dim, clip_model_name_test = load_clip_embeddings(clip_test_embeddings_path)
    if clip_test_dim != clip_input_dim:
        raise ValueError("Dimensiones distintas entre embeddings CLIP de train y test.")
    if clip_model_name is None:
        clip_model_name = clip_model_name_train or clip_model_name_test or "openai/clip-vit-base-patch32"
    clip_projection_dim = clip_projection_dim or 128
    print(f"  Dimensiones: {clip_input_dim} → {clip_projection_dim}")
    print(f"  Modelo CLIP: {clip_model_name}")
else:
    clip_projection_dim = 0

label_names = sorted(train_meta_df["label_name"].unique())
label_to_id = {name: idx for idx, name in enumerate(label_names)}
train_meta_df["label_id"] = train_meta_df["label_name"].map(label_to_id)
test_meta_df["label_id"] = test_meta_df["label_name"].map(label_to_id)

print(f"Train: {len(train_meta_df)} imágenes, Test: {len(test_meta_df)} imágenes.")


# =====================
# 2) Procesador base (ViT)
# =====================
print("\n[2/10] Cargando procesador de imágenes del modelo base...")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
print("Procesador cargado.")


# =====================
# 3) Transformaciones con Albumentations
# =====================
print("\n[3/10] Definiendo transformaciones de entrenamiento y validación...")

train_transforms = A.Compose([
    A.RandomResizedCrop(size=(224, 224), scale=(0.7, 1.0)),  # Más variación en escala
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15, p=0.6),  # Más agresivo
    A.Rotate(limit=20, p=0.6),  # Más rotación
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),  # Adicional
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),  # Ruido gaussiano para regularización
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

test_transforms = A.Compose([
    A.Resize(height=224, width=224),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

print("Transformaciones configuradas correctamente.")


# =====================
# 4) Crear datasets Geo
# =====================
LAT_BINS_TOTAL = NUM_LAT_BINS + 1
LON_BINS_TOTAL = NUM_LON_BINS + 1
print("\n[4/10] Creando datasets personalizados...")
train_ds = GeoImageDataset(
    metadata=train_meta_df,
    root_dir=TRAIN_DIR,
    transforms=train_transforms,
    label_to_id=label_to_id,
    lat_bins_total=LAT_BINS_TOTAL,
    lon_bins_total=LON_BINS_TOTAL,
    clip_embeddings=clip_train_map if use_clip else None,
    clip_dim=clip_input_dim if use_clip else None,
)
test_ds = GeoImageDataset(
    metadata=test_meta_df,
    root_dir=TEST_DIR,
    transforms=test_transforms,
    label_to_id=label_to_id,
    lat_bins_total=LAT_BINS_TOTAL,
    lon_bins_total=LON_BINS_TOTAL,
    clip_embeddings=clip_test_map if use_clip else None,
    clip_dim=clip_input_dim if use_clip else None,
)
print("Datasets creados correctamente.")


# =====================
# 5) Modelo ViT con geolocalización
# =====================
GEO_EMBED_DIM = 32


class GeoViTForImageClassification(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        label_names: List[str],
        num_lat_bins: int,
        num_lon_bins: int,
        geo_embed_dim: int = GEO_EMBED_DIM,
        clip_dim: int = 0,
        clip_embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.backbone = ViTModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.lat_embedding = nn.Embedding(num_lat_bins, geo_embed_dim)
        self.lon_embedding = nn.Embedding(num_lon_bins, geo_embed_dim)
        self.dropout = nn.Dropout(self.backbone.config.hidden_dropout_prob)
        self.dropout_classifier = nn.Dropout(0.3)  # Dropout adicional para reducir overfitting
        self.label_names = label_names
        self.clip_dim = clip_dim
        self.clip_projection = None
        self.clip_embed_dim = clip_embed_dim if clip_dim > 0 else 0
        if clip_dim > 0:
            self.clip_projection = nn.Sequential(
                nn.Linear(clip_dim, self.clip_embed_dim),
                nn.LayerNorm(self.clip_embed_dim),
                nn.GELU(),
                nn.Dropout(self.backbone.config.hidden_dropout_prob),
            )
        fused_dim = hidden_size + geo_embed_dim * 2 + self.clip_embed_dim
        self.classifier = nn.Linear(fused_dim, num_labels)
        self.class_weights = None  # Se asignará después de calcular los pesos
        self.config = self.backbone.config

    def forward(
        self,
        pixel_values: torch.Tensor,
        lat_bins: torch.Tensor,
        lon_bins: torch.Tensor,
        clip_embeddings: Optional[torch.Tensor] = None,
        labels: torch.Tensor | None = None,
    ) -> SequenceClassifierOutput:
        outputs = self.backbone(pixel_values=pixel_values)
        pooled = outputs.pooler_output
        lat_bins = lat_bins.to(pooled.device)
        lon_bins = lon_bins.to(pooled.device)
        lat_emb = self.lat_embedding(lat_bins)
        lon_emb = self.lon_embedding(lon_bins)
        features = [pooled, lat_emb, lon_emb]
        if self.clip_projection is not None:
            if clip_embeddings is None:
                clip_embeddings = torch.zeros(
                    (pixel_values.size(0), self.clip_dim),
                    device=pooled.device,
                    dtype=pooled.dtype,
                )
            else:
                clip_embeddings = clip_embeddings.to(pooled.device, dtype=pooled.dtype)
            clip_feat = self.clip_projection(clip_embeddings)
            features.append(clip_feat)
        fused = torch.cat(features, dim=-1)
        fused = self.dropout(fused)
        fused = self.dropout_classifier(fused)  # Dropout adicional antes del clasificador
        logits = self.classifier(fused)

        loss = None
        if labels is not None:
            if hasattr(self, 'class_weights') and self.class_weights is not None:
                loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            else:
                loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels.to(logits.device))

        return SequenceClassifierOutput(loss=loss, logits=logits)


print("\n[5/10] Cargando modelo base ViT con geofeatures...")
num_labels = len(label_names)
# Nota: class_weights_tensor se asignará después de calcular los pesos en el paso 6
model = GeoViTForImageClassification(
    model_name=MODEL_NAME,
    num_labels=num_labels,
    label_names=label_names,
    num_lat_bins=LAT_BINS_TOTAL,
    num_lon_bins=LON_BINS_TOTAL,
    geo_embed_dim=GEO_EMBED_DIM,
    clip_dim=clip_input_dim if use_clip else 0,
    clip_embed_dim=clip_projection_dim,
)
print(f"Modelo cargado con {num_labels} clases.")
print("Clases detectadas:", label_names)


# =====================
# 6) Calcular pesos y crear sampler
# =====================
print("\n[6/10] Calculando pesos para balancear clases...")
labels = train_meta_df["label_id"].tolist()
label_counts = Counter(labels)
total_count = sum(label_counts.values())

# Calcular pesos base (inverso de frecuencia)
base_weights = {cls: total_count / (len(label_counts) * count) for cls, count in label_counts.items()}

# Identificar clases mayoritarias (que pueden causar sesgo)
majority_classes = []
for cls_idx, name in enumerate(label_names):
    count = label_counts.get(cls_idx, 0)
    if count > total_count * 0.30:  # Más del 30% del dataset
        majority_classes.append((cls_idx, name, count))

weights = base_weights.copy()
if majority_classes:
    print(f"  Detectadas {len(majority_classes)} clases mayoritarias (>30% del dataset):")
    for cls_idx, name, count in majority_classes:
        pct = (count / total_count) * 100
        print(f"    - {name}: {count} imágenes ({pct:.1f}%)")
        # Reducir peso de clases mayoritarias para evitar sesgo
        weights[cls_idx] = base_weights[cls_idx] * 0.8
        print(f"      Peso ajustado: {base_weights[cls_idx]:.3f} → {weights[cls_idx]:.3f} (-20%)")

# Aumentar pesos de clases minoritarias (Africa, Asia)
minority_classes = []
for cls_idx, name in enumerate(label_names):
    if name in ["Africa", "Asia"]:
        count = label_counts.get(cls_idx, 0)
        if count < total_count * 0.15:  # Menos del 15% del dataset
            minority_classes.append((cls_idx, name))

if minority_classes:
    print(f"  Reforzando {len(minority_classes)} clases minoritarias (<15% del dataset):")
    for cls_idx, name in minority_classes:
        count = label_counts.get(cls_idx, 0)
        pct = (count / total_count) * 100
        print(f"    - {name}: {count} imágenes ({pct:.1f}%)")
        # Aumentar peso de clases minoritarias
        weights[cls_idx] = base_weights[cls_idx] * 1.3
        print(f"      Peso ajustado: {base_weights[cls_idx]:.3f} → {weights[cls_idx]:.3f} (+30%)")

sample_weights = [weights[label] for label in labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

# Crear tensor de pesos para la pérdida
class_weights_tensor = torch.tensor([weights.get(cls, 1.0) for cls in range(len(label_names))], dtype=torch.float32)

print("\nSampler creado con pesos balanceados.")
print("Distribución de clases:")
for cls, count in label_counts.items():
    name = label_names[cls]
    pct = (count / total_count) * 100
    print(f" - {name}: {count} imágenes ({pct:.1f}%) - peso: {weights[cls]:.3f}")

# Asignar pesos de clase al modelo para la pérdida
model.class_weights = class_weights_tensor
print(f"\n✓ Pesos de clase asignados al modelo para pérdida ponderada.")


# =====================
# 7) Collate Function
# =====================
def collate_fn(batch):
    """Combina ejemplos individuales en un batch."""
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = torch.tensor([item['labels'] for item in batch])
    lat_bins = torch.tensor([item['lat_bins'] for item in batch])
    lon_bins = torch.tensor([item['lon_bins'] for item in batch])
    collated = {
        'pixel_values': pixel_values,
        'labels': labels,
        'lat_bins': lat_bins,
        'lon_bins': lon_bins,
    }
    if 'clip_embeddings' in batch[0]:
        clip_embeddings = torch.stack([item['clip_embeddings'] for item in batch])
        collated['clip_embeddings'] = clip_embeddings
    return collated


# =====================
# 8) Métricas
# =====================
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {"accuracy": acc, "f1": f1}

# Función para calcular métricas en train (para detectar overfitting)
def compute_train_metrics(trainer):
    """Calcula métricas en el conjunto de entrenamiento para detectar overfitting."""
    print("\n" + "="*60)
    print("EVALUANDO EN CONJUNTO DE ENTRENAMIENTO (detección de overfitting)")
    print("="*60)
    train_metrics = trainer.evaluate(eval_dataset=train_ds)
    print(f"Train Accuracy: {train_metrics.get('eval_accuracy', 0):.4f}")
    print(f"Train F1-Score: {train_metrics.get('eval_f1', 0):.4f}")
    return train_metrics


# =====================
# 9) Configuración del entrenamiento
# =====================
print("\n[7/10] Configurando parámetros de entrenamiento...")

training_args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    num_train_epochs=NUM_EPOCHS,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True if torch.cuda.is_available() else False,
    learning_rate=2e-5,  # Reducido de 3e-5 para entrenamiento más conservador
    weight_decay=0.02,  # Aumentado de 0.01 para más regularización
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    report_to="none",
    dataloader_num_workers=0,  # Para Windows
    remove_unused_columns=False,  # Importante para custom datasets
)

print("Parámetros configurados correctamente.")


# =====================
# 10) Custom Trainer con DataLoader balanceado
# =====================
class BalancedTrainer(Trainer):
    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=0,
        )


print("\n[8/10] Inicializando BalancedTrainer...")
trainer = BalancedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    processing_class=processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)
print("Trainer inicializado correctamente con balanceo de clases.")

trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0005))


# =====================
# 11) Entrenamiento
# =====================
print("\n[9/10] Iniciando entrenamiento...")
print(f"Total de batches por época: {len(train_ds) // BATCH_SIZE}")
print(f"Dispositivo: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
print("="*50)

trainer.train()

print("\n" + "="*50)
print("Entrenamiento finalizado correctamente.")

# Evaluar en train para detectar overfitting
train_metrics = compute_train_metrics(trainer)
test_metrics = trainer.evaluate()

print("\n" + "="*60)
print("COMPARACIÓN TRAIN vs TEST (detección de overfitting)")
print("="*60)
train_acc = train_metrics.get('eval_accuracy', 0)
test_acc = test_metrics.get('eval_accuracy', 0)
train_f1 = train_metrics.get('eval_f1', 0)
test_f1 = test_metrics.get('eval_f1', 0)

print(f"Train Accuracy:  {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"Test Accuracy:   {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Diferencia:      {train_acc - test_acc:.4f} ({(train_acc - test_acc)*100:.2f}%)")
print(f"\nTrain F1-Score:  {train_f1:.4f} ({train_f1*100:.2f}%)")
print(f"Test F1-Score:   {test_f1:.4f} ({test_f1*100:.2f}%)")
print(f"Diferencia:      {train_f1 - test_f1:.4f} ({(train_f1 - test_f1)*100:.2f}%)")

# Advertencia de overfitting
overfitting_threshold = 0.10  # 10% de diferencia
if (train_acc - test_acc) > overfitting_threshold:
    print(f"\n⚠️  ADVERTENCIA: Posible overfitting detectado!")
    print(f"   La diferencia entre train y test accuracy es > {overfitting_threshold*100:.0f}%")
    print(f"   Considera: más regularización, más datos, o menos épocas")
elif (train_acc - test_acc) > 0.05:
    print(f"\n⚠️  Advertencia leve: Diferencia moderada entre train y test")
    print(f"   Monitorea el entrenamiento para evitar overfitting")
else:
    print(f"\n✓ Buen balance entre train y test (sin overfitting aparente)")
print("="*60)

print("\n[10/10] Guardando modelo...")
trainer.save_model(OUT_DIR)
processor.save_pretrained(OUT_DIR)
print(f"Modelo y procesador guardados en: {OUT_DIR}")

best_checkpoint = trainer.state.best_model_checkpoint or str(OUT_DIR)

metadata = {
    "training_mode": TRAINING_MODE,
    "scene_type": SCENE_TYPE if TRAINING_MODE == "scene" else None,
    "train_dir": str(TRAIN_DIR.resolve()),
    "test_dir": str(TEST_DIR.resolve()),
    "train_metadata": str(train_metadata_path.resolve()),
    "test_metadata": str(test_metadata_path.resolve()),
    "label_names": label_names,
    "geo_embed_dim": GEO_EMBED_DIM,
    "model_name": MODEL_NAME,
    "num_lat_bins": LAT_BINS_TOTAL,
    "num_lon_bins": LON_BINS_TOTAL,
    "use_clip_embeddings": use_clip,
    "clip_train_embeddings": str(clip_train_embeddings_path.resolve()) if use_clip else None,
    "clip_test_embeddings": str(clip_test_embeddings_path.resolve()) if use_clip else None,
    "clip_input_dim": clip_input_dim if use_clip else 0,
    "clip_projection_dim": clip_projection_dim if use_clip else 0,
    "clip_model_name": clip_model_name if use_clip else None,
    "checkpoint_dir": best_checkpoint,
}
metadata_path = Path(OUT_DIR) / "training_metadata.json"
metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(f"Metadata de entrenamiento guardada en: {metadata_path}")

