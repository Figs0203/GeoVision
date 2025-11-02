"""
Evaluar el modelo entrenado y generar métricas de rendimiento.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from datasets import load_dataset, DatasetDict
from transformers import ViTForImageClassification, AutoImageProcessor
import torch
from tqdm import tqdm
from PIL import Image


# =====================
# CONFIGURACIÓN
# =====================
DATA_DIR = 'data/images_by_continent'
MODEL_DIR = 'outputs/checkpoints/vit-continent-balanced'  # Ruta completa del modelo guardado
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# =====================
# 1) Cargar dataset
# =====================
print("Cargando dataset...")
raw_ds = load_dataset('imagefolder', data_dir=DATA_DIR)

# Dividir igual que en entrenamiento (80/20)
print("Dividiendo dataset (80/20)...")
raw_ds = raw_ds['train'].train_test_split(test_size=0.2, stratify_by_column='label', seed=42)
dataset = DatasetDict({'train': raw_ds['train'], 'test': raw_ds['test']})

labels = dataset['train'].features['label'].names
print(f"Clases detectadas: {labels}")
print(f"Imágenes de test: {len(dataset['test'])}")


# =====================
# 2) Cargar modelo y procesador
# =====================
print(f"\nCargando modelo desde: {MODEL_DIR}")
print(f"Usando dispositivo: {DEVICE}")

# IMPORTANTE: local_files_only=True para cargar desde disco
model = ViTForImageClassification.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)
model.to(DEVICE)
model.eval()  # Modo evaluación

# Cargar el procesador desde el mismo directorio
try:
    processor = AutoImageProcessor.from_pretrained(
        MODEL_DIR,
        local_files_only=True
    )
    print("Procesador cargado desde el modelo guardado")
except:
    # Fallback: usar el procesador base
    processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    print("Usando procesador base de ViT")


# =====================
# 3) Realizar predicciones
# =====================
print("\nGenerando predicciones...")
y_true, y_pred = [], []

for example in tqdm(dataset['test'], desc="Evaluando"):
    try:
        # Cargar y procesar imagen
        image = example['image']
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            image.load()
            image = image.convert('RGB')
        
        # Procesar imagen
        inputs = processor(images=image, return_tensors='pt')
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # Predicción
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=-1).item()
        
        y_pred.append(pred)
        y_true.append(example['label'])
        
    except Exception as e:
        print(f"Error procesando imagen: {e}")
        continue

print(f"\nPredicciones completadas: {len(y_pred)}/{len(dataset['test'])}")


# =====================
# 4) Matriz de confusión
# =====================
print("\nGenerando matriz de confusión...")
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    xticklabels=labels, 
    yticklabels=labels,
    cmap='Blues',
    cbar_kws={'label': 'Número de predicciones'}
)
plt.xlabel('Predicción', fontsize=12)
plt.ylabel('Etiqueta Real', fontsize=12)
plt.title('Matriz de Confusión - Clasificación por Continente', fontsize=14, fontweight='bold')
plt.tight_layout()

# Guardar figura
output_path = 'outputs/confusion_matrix.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Matriz de confusión guardada en: {output_path}")
plt.show()


# =====================
# 5) Reporte de clasificación
# =====================
print("\n" + "="*60)
print("REPORTE DE CLASIFICACIÓN")
print("="*60)
report = classification_report(
    y_true, 
    y_pred, 
    target_names=labels,
    digits=4
)
print(report)

# Guardar reporte en archivo
report_path = 'outputs/classification_report.txt'
with open(report_path, 'w') as f:
    f.write("REPORTE DE CLASIFICACIÓN - MODELO VIT\n")
    f.write("="*60 + "\n\n")
    f.write(report)
print(f"\nReporte guardado en: {report_path}")


# =====================
# 6) Métricas adicionales
# =====================
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print("\n" + "="*60)
print("MÉTRICAS GENERALES")
print("="*60)
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print("="*60)