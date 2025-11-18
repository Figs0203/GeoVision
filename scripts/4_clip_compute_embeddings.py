"""
Compute CLIP image embeddings for a dataset using metadata CSV.
Saves embeddings in .npz format for later use in training.
Can process both train and test metadata in a single run.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def load_images(paths: List[Path]) -> List[Image.Image]:
    images = []
    for path in paths:
        with Image.open(path) as img:
            images.append(img.convert("RGB"))
    return images


def prompt_path(message: str, default: Path | None = None) -> Path:
    prompt_msg = f"{message}"
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


def ask_yes_no(message: str, default: bool = False) -> bool:
    default_str = "Y/n" if default else "y/N"
    response = input(f"{message} ({default_str}): ").strip().lower()
    if not response:
        return default
    return response in {"y", "yes", "s", "si"}


def compute_embeddings_for_metadata(
    metadata_path: Path,
    root_dir: Path,
    output_path: Path,
    batch_size: int,
    model_name: str,
) -> None:
    """Compute CLIP embeddings for a single metadata CSV."""
    metadata = pd.read_csv(metadata_path)
    if "relative_path" not in metadata.columns:
        raise ValueError("Metadata CSV must contain 'relative_path'.")

    print(f"\nCargando modelo CLIP: {model_name}")
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    model.to(device)
    model.eval()

    embeddings = []
    stored_paths: List[str] = []

    root_dir = Path(root_dir).resolve()
    print(f"\nProcesando {len(metadata)} imágenes desde {metadata_path.name}...")
    
    for start in tqdm(range(0, len(metadata), batch_size), desc=f"Generando embeddings"):
        batch = metadata.iloc[start : start + batch_size]
        image_paths = [root_dir / Path(rel) for rel in batch["relative_path"]]
        images = load_images(image_paths)

        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        image_features = torch.nn.functional.normalize(image_features, dim=-1)
        embeddings.append(image_features.cpu().numpy().astype(np.float32))
        stored_paths.extend(batch["relative_path"].tolist())

    embeddings_array = np.concatenate(embeddings, axis=0)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        embeddings=embeddings_array,
        paths=np.array(stored_paths, dtype=object),
        model_name=np.array(model_name, dtype=object),
    )
    print(f"✓ Guardado: {len(stored_paths)} embeddings en {output_path}")
    print(f"  Dimensión de embeddings: {embeddings_array.shape[1]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute CLIP image embeddings for a dataset using metadata CSV.",
        add_help=False,
    )
    parser.add_argument("--metadata", type=Path, help="CSV with relative_path column.")
    parser.add_argument("--root-dir", type=Path, help="Root directory containing images referenced in metadata.")
    parser.add_argument("--output", type=Path, help="Path to .npz file where embeddings will be stored.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for CLIP inference.")
    parser.add_argument("--model-name", type=str, default="openai/clip-vit-base-patch32", help="CLIP model identifier.")
    parser.add_argument("--train-metadata", type=Path, help="Train metadata CSV (for batch processing).")
    parser.add_argument("--test-metadata", type=Path, help="Test metadata CSV (for batch processing).")
    parser.add_argument("--train-root", type=Path, help="Train images root directory (for batch processing).")
    parser.add_argument("--test-root", type=Path, help="Test images root directory (for batch processing).")
    parser.add_argument("--output-dir", type=Path, help="Output directory for embeddings (for batch processing).")
    parser.add_argument("--help", action="help", help="Mostrar este mensaje y salir.")
    args = parser.parse_args()

    # Modo batch: procesar train y test juntos
    if args.train_metadata and args.test_metadata:
        print("=== Modo batch: Generando embeddings para train y test ===")
        if args.train_root is None or args.test_root is None or args.output_dir is None:
            print("\nConfiguración para modo batch:")
            if args.train_root is None:
                args.train_root = prompt_path("Directorio raíz de imágenes de entrenamiento (p. ej. data/train_outdoor)")
            if args.test_root is None:
                args.test_root = prompt_path("Directorio raíz de imágenes de validación (p. ej. data/test_outdoor)")
            if args.output_dir is None:
                args.output_dir = prompt_path("Directorio de salida para embeddings", Path("data/clip_embeddings"))
        
        # Resolver rutas relativas
        train_meta = (ROOT_DIR / args.train_metadata).resolve() if not args.train_metadata.is_absolute() else args.train_metadata.resolve()
        test_meta = (ROOT_DIR / args.test_metadata).resolve() if not args.test_metadata.is_absolute() else args.test_metadata.resolve()
        train_root = (ROOT_DIR / args.train_root).resolve() if not args.train_root.is_absolute() else args.train_root.resolve()
        test_root = (ROOT_DIR / args.test_root).resolve() if not args.test_root.is_absolute() else args.test_root.resolve()
        output_dir = (ROOT_DIR / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
        
        # Generar nombres de salida basados en los nombres de metadata
        train_output = output_dir / (train_meta.stem + "_embeddings.npz")
        test_output = output_dir / (test_meta.stem + "_embeddings.npz")
        
        print(f"\n[1/2] Procesando embeddings de entrenamiento...")
        compute_embeddings_for_metadata(
            train_meta, train_root, train_output, args.batch_size, args.model_name
        )
        
        print(f"\n[2/2] Procesando embeddings de validación...")
        compute_embeddings_for_metadata(
            test_meta, test_root, test_output, args.batch_size, args.model_name
        )
        
        print(f"\n{'='*60}")
        print("✓ COMPLETADO: Ambos embeddings generados exitosamente")
        print(f"  Entrenamiento: {train_output}")
        print(f"  Validación: {test_output}")
        print(f"{'='*60}")
        return

    # Modo simple: un solo archivo
    metadata_path = args.metadata
    root_dir = args.root_dir
    output_path = args.output

    if metadata_path is None or root_dir is None or output_path is None:
        print("=== Configuración interactiva para embeddings CLIP ===")
        print("\n¿Deseas generar embeddings para train Y test juntos? (recomendado)")
        if ask_yes_no("Modo batch (procesar train y test)", default=True):
            print("\nConfiguración para modo batch:")
            train_meta = prompt_path("Ruta al CSV de metadata de entrenamiento", Path("data/metadata/train_outdoor_metadata.csv"))
            test_meta = prompt_path("Ruta al CSV de metadata de validación", Path("data/metadata/test_outdoor_metadata.csv"))
            train_root = prompt_path("Directorio raíz de imágenes de entrenamiento", Path("data/train_outdoor"))
            test_root = prompt_path("Directorio raíz de imágenes de validación", Path("data/test_outdoor"))
            output_dir = prompt_path("Directorio de salida para embeddings", Path("data/clip_embeddings"))
            
            train_output = output_dir / (train_meta.stem + "_embeddings.npz")
            test_output = output_dir / (test_meta.stem + "_embeddings.npz")
            
            print(f"\n[1/2] Procesando embeddings de entrenamiento...")
            compute_embeddings_for_metadata(
                train_meta, train_root, train_output, args.batch_size, args.model_name
            )
            
            print(f"\n[2/2] Procesando embeddings de validación...")
            compute_embeddings_for_metadata(
                test_meta, test_root, test_output, args.batch_size, args.model_name
            )
            
            print(f"\n{'='*60}")
            print("✓ COMPLETADO: Ambos embeddings generados exitosamente")
            print(f"  Entrenamiento: {train_output}")
            print(f"  Validación: {test_output}")
            print(f"{'='*60}")
            return
        
        # Modo simple: solo un archivo
        if metadata_path is None:
            metadata_path = prompt_path("Ruta al CSV de metadata (p. ej. data/metadata/train_outdoor_metadata.csv)")
        if root_dir is None:
            root_dir = prompt_path("Directorio raíz con imágenes (p. ej. data/train_outdoor)")
        default_output = output_path or Path("data/clip_embeddings") / (Path(metadata_path).stem + "_embeddings.npz")
        output_path = prompt_path("Archivo de salida (.npz)", default_output)
        
        # Resolver rutas relativas
        if not Path(metadata_path).is_absolute():
            metadata_path = ROOT_DIR / metadata_path
        if not Path(root_dir).is_absolute():
            root_dir = ROOT_DIR / root_dir
        if not Path(output_path).is_absolute():
            output_path = ROOT_DIR / output_path
        metadata_path = metadata_path.resolve()
        root_dir = root_dir.resolve()
        output_path = output_path.resolve()
    
    compute_embeddings_for_metadata(
        metadata_path, root_dir, output_path, args.batch_size, args.model_name
    )


if __name__ == "__main__":
    main()

