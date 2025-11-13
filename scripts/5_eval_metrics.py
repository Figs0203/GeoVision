"""
Evaluar el modelo entrenado y generar métricas de rendimiento para la división
train/test derivada del dataset por escenas.
"""

import os
from pathlib import Path

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder
from datasets import DatasetDict, load_dataset
from transformers import AutoImageProcessor, ViTForImageClassification
import torch
from tqdm import tqdm
from PIL import Image


# =====================
parser = argparse.ArgumentParser(
    description="Evaluar el modelo entrenado y generar métricas de rendimiento reales."
)
parser.add_argument(
    "--model-dir",
    type=Path,
    help="Directorio del modelo a evaluar. Si no se especifica, se usa el checkpoint más reciente con metadata.",
)
parser.add_argument(
    "--no-collapse-scene",
    action="store_true",
    help="Si se especifica, no colapsa las etiquetas scene-continent en continentes.",
)
args = parser.parse_args()


def _find_latest_model_dir() -> Path:
    base = Path("outputs/checkpoints")
    if not base.exists():
        raise FileNotFoundError("No se encontró el directorio outputs/checkpoints.")
    candidates = sorted(
        [d for d in base.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if (candidate / "training_metadata.json").exists():
            return candidate
    raise FileNotFoundError(
        "No se encontró metadata de entrenamiento. Especifica --model-dir."
    )


MODEL_DIR = (args.model_dir or _find_latest_model_dir()).resolve()
metadata_path = MODEL_DIR / "training_metadata.json"
if not metadata_path.exists():
    raise FileNotFoundError(
        f"No se encontró training_metadata.json en {MODEL_DIR}. "
        "Ejecuta el entrenamiento con la versión actualizada del script."
    )

metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
TRAINING_MODE = metadata.get("training_mode", "scene_continent")
SCENE_TYPE = metadata.get("scene_type") or "outdoor"
TRAIN_DIR = Path(metadata.get("train_dir", ""))
TEST_DIR = Path(metadata.get("test_dir", ""))
collapse_scene = not args.no_collapse_scene

if TRAINING_MODE not in {"scene", "scene_continent"}:
    raise ValueError(
        f"Metadata inválida: training_mode={TRAINING_MODE}. Debe ser 'scene' o 'scene_continent'."
    )

if not TRAIN_DIR.exists() or not TEST_DIR.exists():
    raise FileNotFoundError(
        f"No se encontraron los directorios {TRAIN_DIR} y {TEST_DIR} definidos en la metadata."
    )

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs("outputs", exist_ok=True)


# =====================
# 1) Cargar datasets
# =====================
print("Cargando dataset desde splits persistentes...")
print(f"Modo de evaluación: {TRAINING_MODE}")
if not TRAIN_DIR.exists() or not TEST_DIR.exists():
    raise FileNotFoundError(
        f"No se encontraron los directorios {TRAIN_DIR} y {TEST_DIR}. "
        "Ejecuta 3_prepare_scene_dataset.py para generarlos."
    )

train_ds = load_dataset("imagefolder", data_dir=str(TRAIN_DIR))["train"]
test_ds = load_dataset("imagefolder", data_dir=str(TEST_DIR))["train"]

label_names = train_ds.features["label"].names
print(f"Clases detectadas: {label_names}")
print(f"Imágenes de test: {len(test_ds)}")


# =====================
# 2) Cargar modelo y procesador
# =====================
print(f"\nCargando modelo desde: {MODEL_DIR}")
print(f"Usando dispositivo: {DEVICE}")

model = ViTForImageClassification.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)
model.to(DEVICE)
model.eval()

try:
    processor = AutoImageProcessor.from_pretrained(
        MODEL_DIR,
        local_files_only=True
    )
    print("Procesador cargado desde el modelo guardado.")
except Exception:
    processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    print("Usando procesador base de ViT.")


# =====================
# 3) Realizar predicciones
# =====================
print("\nGenerando predicciones en conjunto de prueba...")
y_true_ids, y_pred_ids = [], []

for example in tqdm(test_ds, desc="Evaluando"):
    try:
        image = example["image"]
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")

        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=-1).item()

        y_pred_ids.append(pred)
        y_true_ids.append(example["label"])

    except Exception as e:
        print(f"Error procesando imagen: {e}")
        continue

print(f"\nPredicciones completadas: {len(y_pred_ids)}/{len(test_ds)}")


def collapse_label(label_idx: int) -> str:
    name = label_names[label_idx]
    if collapse_scene and "-" in name:
        return name.split("-", 1)[1]
    return name


collapsed_true = [collapse_label(idx) for idx in y_true_ids]
collapsed_pred = [collapse_label(idx) for idx in y_pred_ids]

encoder = LabelEncoder()
encoder.fit(collapsed_true + collapsed_pred)
y_true = encoder.transform(collapsed_true)
y_pred = encoder.transform(collapsed_pred)
collapsed_labels = encoder.classes_


# =====================
# 4) Matriz de confusión
# =====================
print("\nGenerando matriz de confusión...")
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=collapsed_labels,
    yticklabels=collapsed_labels,
    cmap="Blues",
    cbar_kws={"label": "Número de predicciones"}
)
plt.xlabel("Predicción", fontsize=12)
plt.ylabel("Etiqueta real", fontsize=12)
title_suffix = " (colapsada)" if collapse_scene else ""
plt.title(
    f"Matriz de Confusión - Clasificación por Continente{title_suffix}",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()

output_path = "outputs/confusion_matrix.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Matriz de confusión guardada en: {output_path}")
plt.show()


# =====================
# 5) Reporte de clasificación
# =====================
print("\n" + "=" * 60)
print("REPORTE DE CLASIFICACIÓN")
print("=" * 60)
report = classification_report(
    y_true,
    y_pred,
    target_names=collapsed_labels,
    digits=4
)
print(report)

report_path = "outputs/classification_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("REPORTE DE CLASIFICACIÓN - MODELO VIT\n")
    f.write("=" * 60 + "\n\n")
    f.write(report)
print(f"\nReporte guardado en: {report_path}")


# =====================
# 6) Métricas adicionales
# =====================
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted")
recall = recall_score(y_true, y_pred, average="weighted")
f1 = f1_score(y_true, y_pred, average="weighted")

print("\n" + "=" * 60)
print("MÉTRICAS GENERALES")
print("=" * 60)
print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print("=" * 60)
