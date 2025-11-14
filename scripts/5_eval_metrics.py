"""
Evaluar el modelo entrenado y generar métricas de rendimiento reales.
Usa metadatos generados en 3_prepare_scene_dataset.py y el checkpoint guardado.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder
from transformers import ViTModel
from transformers.modeling_outputs import SequenceClassifierOutput

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from preprocessing.geo_dataset import (
    GeoImageDataset,
    LAT_BIN_SIZE,
    LON_BIN_SIZE,
    NUM_LAT_BINS,
    NUM_LON_BINS,
)


class GeoViTForImageClassification(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        num_lat_bins: int,
        num_lon_bins: int,
        geo_embed_dim: int = 32,
    ) -> None:
        super().__init__()
        self.backbone = ViTModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.lat_embedding = nn.Embedding(num_lat_bins, geo_embed_dim)
        self.lon_embedding = nn.Embedding(num_lon_bins, geo_embed_dim)
        self.dropout = nn.Dropout(self.backbone.config.hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size + geo_embed_dim * 2, num_labels)
        self.config = self.backbone.config

    def forward(
        self,
        pixel_values: torch.Tensor,
        lat_bins: torch.Tensor,
        lon_bins: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> SequenceClassifierOutput:
        outputs = self.backbone(pixel_values=pixel_values)
        pooled = outputs.pooler_output
        lat_emb = self.lat_embedding(lat_bins)
        lon_emb = self.lon_embedding(lon_bins)
        fused = torch.cat([pooled, lat_emb, lon_emb], dim=-1)
        fused = self.dropout(fused)
        logits = self.classifier(fused)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


def collate_fn(batch: List[dict]) -> dict:
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
    lat_bins = torch.tensor([item["lat_bins"] for item in batch], dtype=torch.long)
    lon_bins = torch.tensor([item["lon_bins"] for item in batch], dtype=torch.long)
    return {
        "pixel_values": pixel_values,
        "labels": labels,
        "lat_bins": lat_bins,
        "lon_bins": lon_bins,
    }


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


def load_metadata(model_dir: Path) -> dict:
    metadata_path = model_dir / "training_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No se encontró training_metadata.json en {model_dir}. "
            "Ejecuta el entrenamiento con la versión actualizada del script."
        )
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluar el modelo entrenado y generar métricas de rendimiento reales."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Directorio del modelo a evaluar. Si no se especifica, se utiliza el checkpoint más reciente con metadata.",
    )
    parser.add_argument(
        "--no-collapse-scene",
        action="store_true",
        help="Si se especifica, no colapsa las etiquetas scene-continent en continentes.",
    )
    args = parser.parse_args()

    model_dir = (args.model_dir or _find_latest_model_dir()).resolve()
    metadata = load_metadata(model_dir)

    train_metadata_path = Path(metadata.get("train_metadata", ""))
    test_metadata_path = Path(metadata.get("test_metadata", ""))
    train_dir = Path(metadata.get("train_dir", ""))
    test_dir = Path(metadata.get("test_dir", ""))
    label_names = metadata.get("label_names", [])
    geo_embed_dim = metadata.get("geo_embed_dim", 32)
    model_name = metadata.get("model_name", "google/vit-base-patch16-224-in21k")

    if not train_metadata_path.exists() or not test_metadata_path.exists():
        raise FileNotFoundError(
            "No se encontraron metadatos de train/test. Ejecuta 3_prepare_scene_dataset.py nuevamente."
        )

    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"No se encontraron los directorios {train_dir} o {test_dir}. "
            "Ejecuta 3_prepare_scene_dataset.py para generarlos."
        )

    collapse_scene = not args.no_collapse_scene
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("outputs", exist_ok=True)

    test_meta_df = pd.read_csv(test_metadata_path)
    label_to_id = {name: idx for idx, name in enumerate(label_names)}
    test_meta_df["label_id"] = test_meta_df["label_name"].map(label_to_id)

    print(f"Clases detectadas: {label_names}")
    print(f"Imágenes de test: {len(test_meta_df)}")

    test_transforms = A.Compose(
        [
            A.Resize(height=224, width=224),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]
    )

    lat_bins_total = metadata.get("num_lat_bins", NUM_LAT_BINS + 1)
    lon_bins_total = metadata.get("num_lon_bins", NUM_LON_BINS + 1)

    test_dataset = GeoImageDataset(
        metadata=test_meta_df,
        root_dir=test_dir,
        transforms=test_transforms,
        label_to_id=label_to_id,
        lat_bins_total=lat_bins_total,
        lon_bins_total=lon_bins_total,
    )

    print(f"\nCargando modelo desde: {model_dir}")
    print(f"Usando dispositivo: {device}")

    checkpoint_dir = Path(metadata.get("checkpoint_dir", str(model_dir)))
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = (model_dir / checkpoint_dir).resolve()
    if not checkpoint_dir.exists():
        candidates = sorted(model_dir.glob("checkpoint-*/"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError(
                "No se encontró un checkpoint dentro del directorio especificado."
            )
        checkpoint_dir = candidates[0]

    weight_file = checkpoint_dir / "pytorch_model.bin"
    safetensor_file = checkpoint_dir / "model.safetensors"
    if weight_file.exists():
        state_dict = torch.load(weight_file, map_location="cpu")
    elif safetensor_file.exists():
        from safetensors.torch import load_file

        state_dict = load_file(safetensor_file)
    elif (model_dir / "model.safetensors").exists():
        from safetensors.torch import load_file

        state_dict = load_file(model_dir / "model.safetensors")
        checkpoint_dir = model_dir
    else:
        raise FileNotFoundError(
            f"No se encontraron pesos (pytorch_model.bin o model.safetensors) en {checkpoint_dir}."
        )

    model = GeoViTForImageClassification(
        model_name=model_name,
        num_labels=len(label_names),
        num_lat_bins=lat_bins_total,
        num_lon_bins=lon_bins_total,
        geo_embed_dim=geo_embed_dim,
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    eval_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    print("\nGenerando predicciones en conjunto de prueba...")
    y_true_ids: List[int] = []
    y_pred_ids: List[int] = []

    for batch in tqdm(eval_loader, desc="Evaluando"):
        pixel_values = batch["pixel_values"].to(device)
        lat_bins = batch["lat_bins"].to(device)
        lon_bins = batch["lon_bins"].to(device)
        labels = batch["labels"]

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, lat_bins=lat_bins, lon_bins=lon_bins)
            preds = torch.argmax(outputs.logits, dim=-1).cpu()

        y_pred_ids.extend(preds.tolist())
        y_true_ids.extend(labels.tolist())

    print(f"\nPredicciones completadas: {len(y_pred_ids)}/{len(test_dataset)}")

    def collapse_label(name: str) -> str:
        if collapse_scene and "-" in name:
            return name.split("-", 1)[1]
        return name

    collapsed_true = [collapse_label(label_names[idx]) for idx in y_true_ids]
    collapsed_pred = [collapse_label(label_names[idx]) for idx in y_pred_ids]

    encoder = LabelEncoder()
    encoder.fit(collapsed_true + collapsed_pred)
    y_true = encoder.transform(collapsed_true)
    y_pred = encoder.transform(collapsed_pred)
    collapsed_labels = encoder.classes_

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
        cbar_kws={"label": "Número de predicciones"},
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

    print("\n" + "=" * 60)
    print("REPORTE DE CLASIFICACIÓN")
    print("=" * 60)
    report = classification_report(
        y_true,
        y_pred,
        target_names=collapsed_labels,
        digits=4,
    )
    print(report)

    report_path = "outputs/classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("REPORTE DE CLASIFICACIÓN - MODELO VIT\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    print(f"\nReporte guardado en: {report_path}")

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


if __name__ == "__main__":
    main()