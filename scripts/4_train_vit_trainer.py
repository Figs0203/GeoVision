"""
Fine-tune ViT using Hugging Face Trainer on folder-structured images,
with Albumentations augmentations and real class balancing via WeightedRandomSampler.
"""

import argparse
import json
import os
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoImageProcessor,
    TrainingArguments,
    Trainer,
    ViTForImageClassification,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, f1_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


# =====================
# CONFIGURACIÓN GENERAL
# =====================

def _resolve_training_config() -> tuple[str, str]:
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
    args = parser.parse_args()

    training_mode = (args.training_mode or os.getenv("TRAINING_MODE") or "").lower()
    if training_mode not in {"scene", "scene_continent"}:
        print("Seleccione modo de entrenamiento:")
        print("  [1] Solo exteriores/interiores (scene)")
        print("  [2] Ambos (scene_continent)")
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

    return training_mode, scene_type


TRAINING_MODE, SCENE_TYPE = _resolve_training_config()

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

train_ds_hf = load_dataset("imagefolder", data_dir=str(TRAIN_DIR))["train"]
test_ds_hf = load_dataset("imagefolder", data_dir=str(TEST_DIR))["train"]
raw_ds = DatasetDict({"train": train_ds_hf, "test": test_ds_hf})
print(f"Train: {len(train_ds_hf)} imágenes, Test: {len(test_ds_hf)} imágenes.")


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
    A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.Rotate(limit=15, p=0.5),
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
# 4) Custom PyTorch Dataset
# =====================
class ImageDataset(Dataset):
    """Dataset personalizado que procesa imágenes on-the-fly."""
    
    def __init__(self, hf_dataset, transforms):
        self.dataset = hf_dataset
        self.transforms = transforms
        
        # Extraer todas las rutas e índices al inicio
        self.samples = []
        print(f"Preparando índice de {len(hf_dataset)} imágenes...")
        for i in range(len(hf_dataset)):
            example = hf_dataset[i]
            self.samples.append({
                'image': example['image'],
                'label': example['label']
            })
        print(f"Índice creado: {len(self.samples)} imágenes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            # Cargar imagen
            image_obj = sample['image']
            if isinstance(image_obj, str):
                with Image.open(image_obj) as img:
                    image = img.convert("RGB").copy()
            elif isinstance(image_obj, Image.Image):
                image_obj.load()
                image = image_obj.convert("RGB").copy()
            else:
                image = Image.fromarray(np.uint8(image_obj)).convert("RGB")
            
            # Aplicar transformaciones
            np_image = np.array(image)
            aug = self.transforms(image=np_image)
            
            return {
                'pixel_values': aug['image'],
                'labels': sample['label']
            }
        except Exception as e:
            print(f"Error en imagen {idx}: {e}")
            # Retornar una imagen en negro en caso de error
            return {
                'pixel_values': torch.zeros(3, 224, 224),
                'labels': sample['label']
            }


print("\n[4/10] Creando datasets personalizados...")
train_ds = ImageDataset(raw_ds['train'], train_transforms)
test_ds = ImageDataset(raw_ds['test'], test_transforms)
print("Datasets creados correctamente.")


# =====================
# 5) Modelo ViT
# =====================
print("\n[5/10] Cargando modelo base ViT...")
num_labels = len(raw_ds['train'].features['label'].names)
model = ViTForImageClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
print(f"Modelo cargado con {num_labels} clases.")
print("Clases detectadas:", raw_ds['train'].features['label'].names)


# =====================
# 6) Calcular pesos y crear sampler
# =====================
print("\n[6/10] Calculando pesos para balancear clases...")
labels = [sample['label'] for sample in train_ds.samples]
label_counts = Counter(labels)
total_count = sum(label_counts.values())

# peso = total_imágenes / (num_clases * imágenes_de_esa_clase)
weights = {cls: total_count / (len(label_counts) * count) for cls, count in label_counts.items()}
sample_weights = [weights[label] for label in labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

print("Sampler creado con pesos balanceados.")
print("Distribución de clases:")
for cls, count in label_counts.items():
    name = raw_ds['train'].features['label'].names[cls]
    print(f" - {name}: {count} imágenes (peso {weights[cls]:.3f})")


# =====================
# 7) Collate Function
# =====================
def collate_fn(batch):
    """Combina ejemplos individuales en un batch."""
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = torch.tensor([item['labels'] for item in batch])
    return {
        'pixel_values': pixel_values,
        'labels': labels
    }


# =====================
# 8) Métricas
# =====================
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {"accuracy": acc, "f1": f1}


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
    learning_rate=3e-5,
    weight_decay=0.01,
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

trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.001))


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

print("\n[10/10] Guardando modelo...")
trainer.save_model(OUT_DIR)
processor.save_pretrained(OUT_DIR)
print(f"Modelo y procesador guardados en: {OUT_DIR}")

metadata = {
    "training_mode": TRAINING_MODE,
    "scene_type": SCENE_TYPE if TRAINING_MODE == "scene" else None,
    "train_dir": str(TRAIN_DIR),
    "test_dir": str(TEST_DIR),
    "checkpoint_dir": str(OUT_DIR),
}
metadata_path = Path(OUT_DIR) / "training_metadata.json"
metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(f"Metadata de entrenamiento guardada en: {metadata_path}")