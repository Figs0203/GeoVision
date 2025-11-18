"""
Script para verificar que no hay data leakage entre train y test.
Verifica:
1. Duplicados por nombre de archivo
2. Duplicados por hash de imagen (mismo contenido, diferente nombre)
3. Distribución de clases en train vs test
"""

import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def compute_image_hash(image_path: Path) -> str:
    """Calcula hash MD5 del contenido de la imagen."""
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def verify_split(
    train_dir: Path,
    test_dir: Path,
    train_metadata: Path = None,
    test_metadata: Path = None,
) -> Tuple[bool, Dict]:
    """Verifica que no haya data leakage entre train y test."""
    results = {
        "train_files": set(),
        "test_files": set(),
        "train_hashes": {},
        "test_hashes": {},
        "name_overlap": set(),
        "hash_overlap": set(),
        "train_distribution": {},
        "test_distribution": {},
    }

    print(f"\n{'='*60}")
    print("VERIFICACIÓN DE DATA LEAKAGE")
    print(f"{'='*60}\n")

    # 1. Verificar por nombre de archivo
    print("[1/4] Verificando duplicados por nombre de archivo...")
    if train_dir.exists():
        for img_path in train_dir.rglob("*.jpg"):
            results["train_files"].add(img_path.name)
        for img_path in train_dir.rglob("*.jpeg"):
            results["train_files"].add(img_path.name)
        for img_path in train_dir.rglob("*.png"):
            results["train_files"].add(img_path.name)
        print(f"  Train: {len(results['train_files'])} archivos únicos")

    if test_dir.exists():
        for img_path in test_dir.rglob("*.jpg"):
            results["test_files"].add(img_path.name)
        for img_path in test_dir.rglob("*.jpeg"):
            results["test_files"].add(img_path.name)
        for img_path in test_dir.rglob("*.png"):
            results["test_files"].add(img_path.name)
        print(f"  Test:  {len(results['test_files'])} archivos únicos")

    results["name_overlap"] = results["train_files"] & results["test_files"]
    if results["name_overlap"]:
        print(f"  ❌ ERROR: {len(results['name_overlap'])} archivos duplicados por nombre!")
        print(f"     Ejemplos: {list(results['name_overlap'])[:10]}")
        return False, results
    else:
        print(f"  ✓ Sin duplicados por nombre")

    # 2. Verificar por hash de imagen (mismo contenido, diferente nombre)
    print("\n[2/4] Verificando duplicados por contenido (hash MD5)...")
    if train_dir.exists():
        train_images = list(train_dir.rglob("*.jpg")) + list(train_dir.rglob("*.jpeg")) + list(train_dir.rglob("*.png"))
        for img_path in tqdm(train_images, desc="  Procesando train"):
            try:
                img_hash = compute_image_hash(img_path)
                results["train_hashes"][img_hash] = img_path.name
            except Exception as e:
                print(f"  ⚠️  Error procesando {img_path}: {e}")
        print(f"  Train: {len(results['train_hashes'])} imágenes únicas")

    if test_dir.exists():
        test_images = list(test_dir.rglob("*.jpg")) + list(test_dir.rglob("*.jpeg")) + list(test_dir.rglob("*.png"))
        for img_path in tqdm(test_images, desc="  Procesando test"):
            try:
                img_hash = compute_image_hash(img_path)
                results["test_hashes"][img_hash] = img_path.name
            except Exception as e:
                print(f"  ⚠️  Error procesando {img_path}: {e}")
        print(f"  Test:  {len(results['test_hashes'])} imágenes únicas")

    train_hash_set = set(results["train_hashes"].keys())
    test_hash_set = set(results["test_hashes"].keys())
    results["hash_overlap"] = train_hash_set & test_hash_set

    if results["hash_overlap"]:
        print(f"  ❌ ERROR: {len(results['hash_overlap'])} imágenes duplicadas por contenido!")
        print(f"     Mismo contenido, diferentes nombres:")
        for img_hash in list(results["hash_overlap"])[:5]:
            print(f"       Train: {results['train_hashes'][img_hash]}")
            print(f"       Test:  {results['test_hashes'][img_hash]}")
        return False, results
    else:
        print(f"  ✓ Sin duplicados por contenido")

    # 3. Verificar distribución de clases
    print("\n[3/4] Verificando distribución de clases...")
    if train_metadata and train_metadata.exists():
        train_df = pd.read_csv(train_metadata)
        if "label_name" in train_df.columns:
            results["train_distribution"] = train_df["label_name"].value_counts().to_dict()
            print(f"  Train distribución:")
            for label, count in sorted(results["train_distribution"].items()):
                print(f"    {label}: {count}")

    if test_metadata and test_metadata.exists():
        test_df = pd.read_csv(test_metadata)
        if "label_name" in test_df.columns:
            results["test_distribution"] = test_df["label_name"].value_counts().to_dict()
            print(f"  Test distribución:")
            for label, count in sorted(results["test_distribution"].items()):
                print(f"    {label}: {count}")

    # 4. Resumen final
    print("\n[4/4] Resumen de verificación:")
    print(f"  ✓ Archivos en train: {len(results['train_files'])}")
    print(f"  ✓ Archivos en test:  {len(results['test_files'])}")
    print(f"  ✓ Imágenes únicas en train: {len(results['train_hashes'])}")
    print(f"  ✓ Imágenes únicas en test:  {len(results['test_hashes'])}")
    print(f"  ✓ Duplicados por nombre: {len(results['name_overlap'])}")
    print(f"  ✓ Duplicados por contenido: {len(results['hash_overlap'])}")

    print(f"\n{'='*60}")
    if results["name_overlap"] or results["hash_overlap"]:
        print("❌ DATA LEAKAGE DETECTADO - CORRIGE ANTES DE ENTRENAR")
        print(f"{'='*60}")
        return False, results
    else:
        print("✓ VERIFICACIÓN EXITOSA - Sin data leakage detectado")
        print(f"{'='*60}")
        return True, results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Verificar data leakage entre train y test")
    parser.add_argument(
        "--train-dir",
        type=Path,
        help="Directorio de entrenamiento (p. ej. data/train_outdoor)",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        help="Directorio de test (p. ej. data/test_outdoor)",
    )
    parser.add_argument(
        "--train-metadata",
        type=Path,
        help="CSV de metadata de entrenamiento (opcional)",
    )
    parser.add_argument(
        "--test-metadata",
        type=Path,
        help="CSV de metadata de test (opcional)",
    )
    args = parser.parse_args()

    if not args.train_dir or not args.test_dir:
        print("Uso: python scripts/verify_data_leakage.py --train-dir <dir> --test-dir <dir>")
        print("\nEjemplo:")
        print("  python scripts/verify_data_leakage.py \\")
        print("    --train-dir data/train_outdoor \\")
        print("    --test-dir data/test_outdoor \\")
        print("    --train-metadata data/metadata/train_outdoor_metadata.csv \\")
        print("    --test-metadata data/metadata/test_outdoor_metadata.csv")
        return

    train_dir = (ROOT_DIR / args.train_dir).resolve() if not args.train_dir.is_absolute() else args.train_dir.resolve()
    test_dir = (ROOT_DIR / args.test_dir).resolve() if not args.test_dir.is_absolute() else args.test_dir.resolve()
    train_meta = (
        (ROOT_DIR / args.train_metadata).resolve()
        if args.train_metadata and not args.train_metadata.is_absolute()
        else args.train_metadata.resolve() if args.train_metadata else None
    )
    test_meta = (
        (ROOT_DIR / args.test_metadata).resolve()
        if args.test_metadata and not args.test_metadata.is_absolute()
        else args.test_metadata.resolve() if args.test_metadata else None
    )

    is_valid, results = verify_split(train_dir, test_dir, train_meta, test_meta)

    if not is_valid:
        sys.exit(1)


if __name__ == "__main__":
    main()

