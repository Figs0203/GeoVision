"""
Crea estructura de carpetas data/images_by_continent/<continent>/
y soporte incremental: copia solo imágenes nuevas salvo que se pida full-refresh.
También puede dividir en train/test si se requiere.
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Organiza imágenes por continente con soporte incremental."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("coords_with_continent.csv"),
        help="CSV con columnas filename y continent.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("images"),
        help="Directorio de origen con imágenes descargadas.",
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=Path("data/images_by_continent"),
        help="Directorio destino organizado por continente.",
    )
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Eliminar destino y copiar todas las imágenes desde cero.",
    )
    parser.add_argument(
        "--create-split",
        action="store_true",
        help="Generar carpetas data/train y data/test (80/20).",
    )
    parser.add_argument(
        "--split-test-size",
        type=float,
        default=0.2,
        help="Proporción del conjunto para test.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Semilla para el split estratificado.",
    )
    return parser.parse_args()


def normalize_continent(continent: str) -> str:
    """Normaliza nombres de continentes."""
    if continent in {"North America", "South America"}:
        return "Americas"
    if continent == "Antarctica":
        return "Antarctica"
    return continent


def main() -> None:
    args = parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"No se encontró el CSV: {args.csv}")

    if not args.source_dir.exists():
        raise FileNotFoundError(f"No se encontró el directorio de origen: {args.source_dir}")

    # Cargar CSV
    print(f"Cargando CSV desde: {args.csv}")
    df = pd.read_csv(args.csv)

    if "filename" not in df.columns or "continent" not in df.columns:
        raise ValueError("El CSV debe contener las columnas 'filename' y 'continent'.")

    # Normalizar continentes
    df["continent"] = df["continent"].map(normalize_continent)

    # Filtrar Unknown
    df = df[df["continent"] != "Unknown"].copy()
    print(f"Total de imágenes a procesar: {len(df)}")

    # Crear directorio destino
    dest_dir = Path(args.dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Full refresh: eliminar destino
    if args.full_refresh:
        print("Modo full-refresh: eliminando destino y copiando todas las imágenes...")
        if dest_dir.exists():
            for item in dest_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

    # Copiar imágenes por continente
    copied_count = 0
    skipped_count = 0

    print("\nOrganizando imágenes por continente...")
    for continent in df["continent"].unique():
        continent_dir = dest_dir / continent
        continent_dir.mkdir(exist_ok=True)

        continent_images = df[df["continent"] == continent]
        print(f"\n  {continent}: {len(continent_images)} imágenes")

        for _, row in continent_images.iterrows():
            src_path = args.source_dir / row["filename"]
            dst_path = continent_dir / row["filename"]

            if not src_path.exists():
                print(f"    Advertencia: No se encontró {src_path}")
                continue

            if dst_path.exists() and not args.full_refresh:
                skipped_count += 1
                continue

            try:
                shutil.copy2(src_path, dst_path)
                copied_count += 1
            except Exception as e:
                print(f"    Error copiando {src_path}: {e}")

    print(f"\n{'='*60}")
    print(f"RESUMEN:")
    print(f"  Imágenes copiadas: {copied_count}")
    print(f"  Imágenes omitidas (ya existían): {skipped_count}")
    print(f"  Total procesadas: {copied_count + skipped_count}")

    # Crear splits si se solicita
    if args.create_split:
        print("\nGenerando splits train/test...")
        train_dir = Path("data/train")
        test_dir = Path("data/test")
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        # Split estratificado por continente
        train_df, test_df = train_test_split(
            df,
            test_size=args.split_test_size,
            random_state=args.split_seed,
            stratify=df["continent"],
        )

        print(f"  Train: {len(train_df)} imágenes")
        print(f"  Test: {len(test_df)} imágenes")

        # Copiar a train y test
        for split_name, split_df in [("train", train_df), ("test", test_df)]:
            split_path = Path(f"data/{split_name}")
            for continent in split_df["continent"].unique():
                continent_dir = split_path / continent
                continent_dir.mkdir(exist_ok=True)

                continent_images = split_df[split_df["continent"] == continent]
                for _, row in continent_images.iterrows():
                    src_path = dest_dir / continent / row["filename"]
                    dst_path = continent_dir / row["filename"]
                    if src_path.exists() and not dst_path.exists():
                        shutil.copy2(src_path, dst_path)

        print(f"\nSplits creados en data/train/ y data/test/")


if __name__ == "__main__":
    main()

