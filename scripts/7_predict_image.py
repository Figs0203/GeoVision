# scripts/7_predict_image.py
"""
Predice el continente de una imagen usando el modelo ViT entrenado.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoImageProcessor, CLIPModel, CLIPProcessor, ViTModel

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from preprocessing.geo_dataset import (
    NUM_LAT_BINS,
    NUM_LON_BINS,
    compute_lat_bin,
    compute_lon_bin,
)


def load_metadata(model_dir: Path) -> dict:
    metadata_path = model_dir / "training_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No se encontró training_metadata.json en {model_dir}. Ejecuta el entrenamiento actualizado."
        )
    return json.loads(metadata_path.read_text(encoding="utf-8"))


class GeoViTForImageClassification(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        num_lat_bins: int,
        num_lon_bins: int,
        geo_embed_dim: int = 32,
        clip_dim: int = 0,
        clip_embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.backbone = ViTModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.lat_embedding = nn.Embedding(num_lat_bins, geo_embed_dim)
        self.lon_embedding = nn.Embedding(num_lon_bins, geo_embed_dim)
        self.dropout = nn.Dropout(self.backbone.config.hidden_dropout_prob)
        self.clip_dim = clip_dim
        self.clip_embed_dim = clip_embed_dim if clip_dim > 0 else 0
        self.clip_projection = None
        if clip_dim > 0:
            self.clip_projection = nn.Sequential(
                nn.Linear(clip_dim, self.clip_embed_dim),
                nn.LayerNorm(self.clip_embed_dim),
                nn.GELU(),
                nn.Dropout(self.backbone.config.hidden_dropout_prob),
            )
        fused_dim = hidden_size + geo_embed_dim * 2 + self.clip_embed_dim
        self.classifier = nn.Linear(fused_dim, num_labels)
        self.config = self.backbone.config

    def forward(
        self,
        pixel_values: torch.Tensor,
        lat_bins: torch.Tensor,
        lon_bins: torch.Tensor,
        clip_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
        logits = self.classifier(fused)
        return logits


def find_checkpoint_dir(model_dir: Path, metadata: dict) -> Path:
    checkpoint_dir = Path(metadata.get("checkpoint_dir", str(model_dir)))
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = (model_dir / checkpoint_dir).resolve()
    if checkpoint_dir.exists():
        return checkpoint_dir
    candidates = sorted(model_dir.glob("checkpoint-*/"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    return model_dir


def load_state_dict(checkpoint_dir: Path) -> dict:
    weight_file = checkpoint_dir / "pytorch_model.bin"
    safetensor_file = checkpoint_dir / "model.safetensors"
    if weight_file.exists():
        return torch.load(weight_file, map_location="cpu")
    if safetensor_file.exists():
        from safetensors.torch import load_file

        return load_file(safetensor_file)
    raise FileNotFoundError(
        f"No se encontraron pesos en {checkpoint_dir}."
    )


def compute_clip_embedding(
    image: Image.Image,
    model_name: str,
    device: torch.device,
) -> torch.Tensor:
    clip_processor = CLIPProcessor.from_pretrained(model_name)
    clip_model = CLIPModel.from_pretrained(model_name)
    clip_model.to(device)
    clip_model.eval()
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    features = torch.nn.functional.normalize(features, dim=-1)
    return features


def predict(
    image_path: str,
    model_dir: str = "outputs/checkpoints/vit-continent-balanced-outdoor",
    show_image: bool = False,
    show_probabilities: bool = False,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
):
    image_path = Path(image_path)
    model_dir_path = Path(model_dir)

    if not image_path.exists():
        raise FileNotFoundError(f"La imagen no existe: {image_path}")
    if not model_dir_path.exists():
        raise FileNotFoundError(f"El modelo no existe: {model_dir}")

    metadata = load_metadata(model_dir_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    label_names = metadata.get("label_names", [])
    if not label_names:
        raise ValueError("La metadata no contiene 'label_names'.")

    geo_embed_dim = int(metadata.get("geo_embed_dim", 32) or 32)
    num_lat_bins = int(metadata.get("num_lat_bins", NUM_LAT_BINS + 1) or (NUM_LAT_BINS + 1))
    num_lon_bins = int(metadata.get("num_lon_bins", NUM_LON_BINS + 1) or (NUM_LON_BINS + 1))
    clip_input_dim = int(metadata.get("clip_input_dim", 0) or 0)
    clip_projection_dim = int(metadata.get("clip_projection_dim", 128) or 0)
    clip_model_name = metadata.get("clip_model_name") or "openai/clip-vit-base-patch32"
    model_name = metadata.get("model_name", "google/vit-base-patch16-224-in21k")

    checkpoint_dir = find_checkpoint_dir(model_dir_path, metadata)
    print(f"Cargando modelo desde: {checkpoint_dir}")
    state_dict = load_state_dict(checkpoint_dir)

    model = GeoViTForImageClassification(
        model_name=model_name,
        num_labels=len(label_names),
        num_lat_bins=num_lat_bins,
        num_lon_bins=num_lon_bins,
        geo_embed_dim=geo_embed_dim,
        clip_dim=clip_input_dim,
        clip_embed_dim=clip_projection_dim,
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    try:
        processor = AutoImageProcessor.from_pretrained(model_dir_path, local_files_only=True)
        print("Procesador cargado desde el modelo guardado.")
    except Exception:
        processor = AutoImageProcessor.from_pretrained(model_name)
        print("Usando procesador base de ViT.")

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    if lat is None:
        lat_idx = num_lat_bins - 1
    else:
        lat_idx = compute_lat_bin(lat)
    if lon is None:
        lon_idx = num_lon_bins - 1
    else:
        lon_idx = compute_lon_bin(lon)
    lat_tensor = torch.tensor([lat_idx], device=device, dtype=torch.long)
    lon_tensor = torch.tensor([lon_idx], device=device, dtype=torch.long)

    clip_tensor = None
    if clip_input_dim > 0:
        clip_features = compute_clip_embedding(image, clip_model_name, device)
        if clip_features.shape[1] != clip_input_dim:
            raise ValueError(
                f"El embedding CLIP obtenido ({clip_features.shape[1]}) no coincide con la dimensión esperada ({clip_input_dim})."
            )
        clip_tensor = clip_features

    with torch.no_grad():
        logits = model(
            pixel_values=pixel_values,
            lat_bins=lat_tensor,
            lon_bins=lon_tensor,
            clip_embeddings=clip_tensor,
        )
        probabilities = torch.softmax(logits, dim=-1)[0].cpu()
        predicted_class = torch.argmax(probabilities).item()

    predicted_label = label_names[predicted_class]
    probs_dict = {label_names[i]: float(probabilities[i]) for i in range(len(label_names))}
    confidence = probs_dict[predicted_label]

    # Obtener las dos mejores predicciones
    sorted_probs = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
    top1_label, top1_conf = sorted_probs[0]
    top2_label, top2_conf = sorted_probs[1] if len(sorted_probs) > 1 else (None, 0.0)

    print("\n" + "=" * 60)
    print(f"PREDICCIÓN: {top1_label}")
    print(f"Confianza: {top1_conf * 100:.2f}%")
    if top2_label:
        print(f"\nSegunda opción: {top2_label}")
        print(f"Confianza: {top2_conf * 100:.2f}%")
    print("=" * 60)

    if show_probabilities:
        print("\nProbabilidades por continente:")
        sorted_probs = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
        for continent, prob in sorted_probs:
            bar = "█" * int(prob * 50)
            print(f"  {continent:12s}: {prob * 100:5.2f}% {bar}")

    if show_image:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax1.imshow(image)
        ax1.axis("off")
        ax1.set_title(f"Imagen Original\n{image_path.name}", fontsize=12)

        sorted_probs = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
        continents = [item[0] for item in sorted_probs]
        probs = [item[1] * 100 for item in sorted_probs]
        colors = ["green" if cont == predicted_label else "steelblue" for cont in continents]

        ax2.barh(continents, probs, color=colors)
        ax2.set_xlabel("Probabilidad (%)", fontsize=11)
        ax2.set_title(
            f"Predicción: {predicted_label}\nConfianza: {confidence * 100:.1f}%",
            fontsize=12,
            fontweight="bold",
        )
        ax2.set_xlim(0, 100)
        for i, (cont, prob) in enumerate(zip(continents, probs)):
            ax2.text(prob + 1, i, f"{prob:.1f}%", va="center", fontsize=9)
        plt.tight_layout()
        plt.show()

    return predicted_label, probs_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predice el continente de una imagen usando el modelo ViT con geofeatures/CLIP."
    )
    parser.add_argument("image", type=str, nargs="?", help="Ruta a la imagen a predecir (argumento posicional)")
    parser.add_argument("--image", type=str, dest="image_flag", help="Ruta a la imagen a predecir (flag alternativo)")
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/checkpoints/vit-continent-balanced-outdoor",
        help="Ruta al directorio del modelo entrenado",
    )
    parser.add_argument("--lat", type=float, help="Latitud aproximada de la imagen (opcional).")
    parser.add_argument("--lon", type=float, help="Longitud aproximada de la imagen (opcional).")
    parser.add_argument("--show", action="store_true", help="Mostrar la imagen con visualización de probabilidades")
    parser.add_argument("--probs", action="store_true", help="Mostrar probabilidades de todas las clases")
    args = parser.parse_args()
    
    # Determinar la ruta de la imagen: usar --image si está presente, sino usar argumento posicional
    image_path = args.image_flag if args.image_flag else args.image
    
    if not image_path:
        parser.error("Debe proporcionar la ruta a la imagen (como argumento posicional o con --image)\n"
                    "Ejemplo: python scripts/7_predict_image.py \"ruta\\con\\espacios\\imagen.jpg\"")
    
    # Resolver la ruta si es relativa
    image_path = Path(image_path)
    if not image_path.is_absolute():
        image_path = ROOT_DIR / image_path
    image_path = image_path.resolve()

    try:
        predict(
            image_path,
            model_dir=args.model,
            show_image=args.show,
            show_probabilities=args.probs,
            lat=args.lat,
            lon=args.lon,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        import traceback

        traceback.print_exc()
        