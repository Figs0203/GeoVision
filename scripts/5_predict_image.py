# scripts/5_predict_image.py
"""
Predice el continente de una imagen usando el modelo ViT entrenado.

Ejemplos de uso:
    python scripts/5_predict_image.py --image data/test_image.jpg
    python scripts/5_predict_image.py --image data/test_image.jpg --model outputs/checkpoints/vit-continent-balanced
    python scripts/5_predict_image.py --image data/test_image.jpg --show
"""

import argparse
import os
from transformers import ViTForImageClassification, AutoImageProcessor
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np


def predict(
    image_path: str, 
    model_dir: str = "outputs/checkpoints/vit-continent-balanced",
    show_image: bool = False,
    show_probabilities: bool = False
):
    """
    Realiza la predicción del continente de una imagen.
    
    Args:
        image_path: Ruta a la imagen
        model_dir: Directorio del modelo entrenado
        show_image: Si True, muestra la imagen con la predicción
        show_probabilities: Si True, muestra las probabilidades de todas las clases
    
    Returns:
        tuple: (predicted_label, probabilities_dict)
    """
    # Validar que existe la imagen
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"La imagen no existe: {image_path}")
    
    # Validar que existe el modelo
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"El modelo no existe: {model_dir}")

    # Detectar dispositivo (GPU o CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo: {device}")

    # Cargar modelo y procesador
    print(f"Cargando modelo desde: {model_dir}...")
    model = ViTForImageClassification.from_pretrained(
        model_dir,
        local_files_only=True  # IMPORTANTE: cargar desde disco local
    )
    model.to(device)
    model.eval()
    
    # Intentar cargar procesador del modelo guardado, sino usar el base
    try:
        processor = AutoImageProcessor.from_pretrained(
            model_dir,
            local_files_only=True
        )
        print("Procesador cargado desde el modelo guardado")
    except:
        processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        print("Usando procesador base de ViT")
    
    # Obtener labels del modelo
    labels = list(model.config.id2label.values())
    
    # Si los labels son genéricos (LABEL_0, LABEL_1...), usar mapeo manual
    if labels[0].startswith('LABEL_'):
        print("Labels genéricos detectados, usando mapeo manual...")
        # Orden basado en el dataset original
        label_map = {
            'LABEL_0': 'Africa',
            'LABEL_1': 'Americas', 
            'LABEL_2': 'Asia',
            'LABEL_3': 'Europe',
            'LABEL_4': 'Oceania'
        }
        labels = [label_map.get(label, label) for label in labels]
    
    print(f"Clases disponibles: {labels}")

    # Cargar y preparar imagen
    print(f"\nProcesando imagen: {image_path}")
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Predicción
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Calcular probabilidades (softmax)
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
        predicted_class = torch.argmax(logits, dim=-1).item()
    
    # Crear diccionario de probabilidades
    probs_dict = {
        labels[i]: float(probabilities[i]) 
        for i in range(len(labels))
    }
    
    predicted_label = labels[predicted_class]
    confidence = probs_dict[predicted_label]
    
    # Mostrar resultados
    print("\n" + "="*60)
    print(f"PREDICCION: {predicted_label}")
    print(f"Confianza: {confidence*100:.2f}%")
    print("="*60)
    
    if show_probabilities:
        print("\nProbabilidades por continente:")
        # Ordenar por probabilidad descendente
        sorted_probs = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
        for continent, prob in sorted_probs:
            bar = "█" * int(prob * 50)
            print(f"  {continent:12s}: {prob*100:5.2f}% {bar}")
    
    # Visualizar imagen con predicción
    if show_image:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Mostrar imagen
        ax1.imshow(image)
        ax1.axis('off')
        ax1.set_title(f"Imagen Original\n{os.path.basename(image_path)}", fontsize=12)
        
        # Mostrar gráfico de probabilidades
        sorted_probs = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
        continents = [item[0] for item in sorted_probs]
        probs = [item[1] * 100 for item in sorted_probs]
        colors = ['green' if cont == predicted_label else 'steelblue' for cont in continents]
        
        ax2.barh(continents, probs, color=colors)
        ax2.set_xlabel('Probabilidad (%)', fontsize=11)
        ax2.set_title(f'Predicción: {predicted_label}\nConfianza: {confidence*100:.1f}%', 
                      fontsize=12, fontweight='bold')
        ax2.set_xlim(0, 100)
        
        # Agregar valores en las barras
        for i, (cont, prob) in enumerate(zip(continents, probs)):
            ax2.text(prob + 1, i, f'{prob:.1f}%', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    return predicted_label, probs_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predice el continente de una imagen usando el modelo ViT entrenado."
    )
    parser.add_argument(
        "--image", 
        type=str, 
        required=True, 
        help="Ruta a la imagen a predecir"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="outputs/checkpoints/vit-continent-balanced",
        help="Ruta al directorio del modelo entrenado"
    )
    parser.add_argument(
        "--show", 
        action="store_true",
        help="Mostrar la imagen con visualización de probabilidades"
    )
    parser.add_argument(
        "--probs", 
        action="store_true",
        help="Mostrar probabilidades de todas las clases"
    )
    
    args = parser.parse_args()

    try:
        predict(
            args.image, 
            model_dir=args.model,
            show_image=args.show,
            show_probabilities=args.probs
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()