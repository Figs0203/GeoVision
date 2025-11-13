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
        help="Proporción para el split test cuando se crea train/test.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Semilla usada para el split train/test.",
    )
    return parser.parse_args()


def normalize_continent(continent: str) -> str:
    if continent in {"North America", "South America"}:
        return "Americas"
    if continent == "Antarctica":
        return "Antarctica"
    return continent


def copy_images(
    df: pd.DataFrame,
    source_dir: Path,
    dest_dir: Path,
    full_refresh: bool,
) -> None:
    if full_refresh and dest_dir.exists():
        print(f"Eliminando destino existente {dest_dir} (full-refresh).")
        shutil.rmtree(dest_dir)

    dest_dir.mkdir(parents=True, exist_ok=True)
    for continent in df["continent"].unique():
        (dest_dir / continent).mkdir(parents=True, exist_ok=True)

    existing = set()
    if dest_dir.exists() and not full_refresh:
        for continent in df["continent"].unique():
            cont_folder = dest_dir / continent
            if cont_folder.exists():
                existing.update(f.name for f in cont_folder.glob("*.*"))
        print(f"Imágenes ya presentes en destino: {len(existing)}")

    missing = []
    copied = 0
    skipped = 0
    for _, row in df.iterrows():
        src = source_dir / row["filename"]
        dst = dest_dir / row["continent"] / row["filename"]
        if not src.exists():
            missing.append(str(src))
            continue
        if not full_refresh and dst.name in existing:
            skipped += 1
            continue
        shutil.copy2(src, dst)
        copied += 1

    print(
        f"Copiado finalizado. Nuevas imágenes copiadas: {copied}, "
        f"omitidas por existir: {skipped}, faltantes: {len(missing)}"
    )
    if missing:
        print("Ejemplos de archivos faltantes:")
        for sample in missing[:10]:
            print(f"  - {sample}")


def create_split(
    dest_dir: Path,
    test_size: float,
    seed: int,
) -> None:
    train_dir = Path("data/train")
    test_dir = Path("data/test")

    for split_dir in (train_dir, test_dir):
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)

    for continent_dir in dest_dir.iterdir():
        if not continent_dir.is_dir():
            continue
        images = sorted(continent_dir.glob("*.jpg"))
        if not images:
            continue
        train_imgs, test_imgs = train_test_split(
            images, test_size=test_size, random_state=seed
        )
        (train_dir / continent_dir.name).mkdir(parents=True, exist_ok=True)
        (test_dir / continent_dir.name).mkdir(parents=True, exist_ok=True)
        for src in train_imgs:
            shutil.copy2(src, train_dir / continent_dir.name / src.name)
        for src in test_imgs:
            shutil.copy2(src, test_dir / continent_dir.name / src.name)

    print("División train/test generada en data/train y data/test.")


def main() -> None:
    args = parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"No se encontró el CSV: {args.csv}")
    if not args.source_dir.exists():
        raise FileNotFoundError(f"No se encontró el directorio de imágenes: {args.source_dir}")

    print("Leyendo CSV con continentes...")
    df = pd.read_csv(args.csv)
    print(f"Total de filas en CSV: {len(df)}")

    df = df[df["continent"] != "Unknown"].copy()
    df["continent"] = df["continent"].map(normalize_continent)
    print(f"Filas con continente válido: {len(df)}")

    copy_images(df, args.source_dir.resolve(), args.dest_dir.resolve(), args.full_refresh)

    print("\nDistribución final por continente:")
    for continent in sorted(df["continent"].unique()):
        folder = args.dest_dir / continent
        count = len(list(folder.glob("*.jpg"))) if folder.exists() else 0
        print(f"  {continent}: {count} imágenes")

    if args.create_split:
        print("\nGenerando división train/test...")
        create_split(args.dest_dir.resolve(), args.split_test_size, args.split_seed)

    print("\nProceso completado.")


if __name__ == "__main__":
    main()
