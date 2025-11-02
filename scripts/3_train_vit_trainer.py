"""
Fine-tune ViT using Hugging Face Trainer on folder-structured images,
with Albumentations augmentations and real class balancing via WeightedRandomSampler.
"""

import os
import numpy as np
from PIL import Image
from datasets import load_dataset, DatasetDict
from transformers import AutoImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader
from collections import Counter


# =====================
# CONFIGURACIÓN GENERAL
# =====================
DATA_DIR = 'data/images_by_continent'
MODEL_NAME = 'google/vit-base-patch16-224-in21k'
OUT_DIR = 'outputs/checkpoints/vit-continent-balanced'
BATCH_SIZE = 16
NUM_EPOCHS = 5


# =====================
# 1) Cargar dataset
# =====================
print("\n[1/10] Cargando dataset desde carpetas...")
raw_ds = load_dataset('imagefolder', data_dir=DATA_DIR)
print("Dataset cargado con las siguientes divisiones:")
print(raw_ds)

print("\nDividiendo en subconjuntos de entrenamiento y prueba (80/20)...")
raw_ds = raw_ds['train'].train_test_split(test_size=0.2, stratify_by_column='label')
raw_ds = DatasetDict({'train': raw_ds['train'], 'test': raw_ds['test']})
print(f"División completada. Train: {len(raw_ds['train'])}, Test: {len(raw_ds['test'])}")


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
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=NUM_EPOCHS,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True if torch.cuda.is_available() else False,
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