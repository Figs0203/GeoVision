import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

VALID_SCENES: Set[str] = {"indoor", "outdoor"}
CONTINENT_NORMALIZATION: Dict[str, str] = {
    "North America": "Americas",
    "South America": "Americas",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reorganiza imágenes según escena (indoor/outdoor) y continente. "
            "Crea estructura destino scene/continent/imagen y genera splits persistentes."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("outputs/scene_predictions.csv"),
        help="CSV generado por run_scene_filter con columnas filename, scene_type, continent.",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        choices=sorted(VALID_SCENES),
        help="Escenas a incluir. Si no se especifica, se pedirá por consola.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/images_by_continent"),
        help="Directorio base donde están las imágenes originales referenciadas en el CSV.",
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=Path("data/images_by_scene"),
        help="Directorio destino para la nueva estructura.",
    )
    parser.add_argument(
        "--split-base",
        type=Path,
        default=Path("data"),
        help="Directorio base donde se crearán los splits persistentes.",
    )
    parser.add_argument(
        "--split-size",
        type=float,
        default=0.2,
        help="Proporción del conjunto para test (por cada escena/continente).",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Semilla para el split estratificado.",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Mover archivos en lugar de copiarlos.",
    )
    parser.add_argument(
        "--clean-dest",
        dest="clean_dest",
        action="store_true",
        help="Vaciar el directorio destino antes de copiar/mover.",
    )
    parser.add_argument(
        "--no-clean-dest",
        dest="clean_dest",
        action="store_false",
        help="Conservar el directorio destino existente.",
    )
    parser.add_argument(
        "--clean-splits",
        dest="clean_splits",
        action="store_true",
        help="Vaciar los directorios de splits antes de generarlos.",
    )
    parser.add_argument(
        "--no-clean-splits",
        dest="clean_splits",
        action="store_false",
        help="Conservar los splits existentes.",
    )
    parser.set_defaults(clean_dest=True, clean_splits=True)
    parser.add_argument(
        "--on-duplicate",
        choices=["skip", "overwrite"],
        default="skip",
        help="Comportamiento cuando el archivo destino ya existe.",
    )
    args = parser.parse_args()

    if not args.scenes:
        print("Seleccione qué escenas incluir en el dataset:")
        print("  [1] Solo outdoor")
        print("  [2] Solo indoor")
        print("  [3] Ambos (outdoor + indoor)")
        option = input("Ingrese opción (1/2/3, default 3): ").strip() or "3"
        option_map = {
            "1": ["outdoor"],
            "2": ["indoor"],
            "3": ["outdoor", "indoor"],
        }
        if option not in option_map:
            raise ValueError(f"Opción no válida '{option}'. Debe ser 1, 2 o 3.")
        args.scenes = option_map[option]
    else:
        args.scenes = [scene.lower() for scene in args.scenes]

    df = pd.read_csv(args.csv)
    _validate_csv(df)

    selected_scenes = set(args.scenes)
    df = df[df["scene_type"].isin(selected_scenes)].copy()
    if df.empty:
        raise ValueError(
            f"No hay filas para las escenas seleccionadas {selected_scenes} en el CSV."
        )

    df["continent"] = df["continent"].map(_normalize_continent)
    print("Distribución por escena y continente:")
    print(df.groupby(["scene_type", "continent"]).size())

    scene_dir = args.dest_dir.resolve()
    if args.clean_dest and scene_dir.exists():
        print(f"Limpiando directorio destino {scene_dir}")
        shutil.rmtree(scene_dir)
    scene_dir.mkdir(parents=True, exist_ok=True)

    missing_files = []
    copied = 0
    skipped = 0
    duplicates = 0

    source_dir = args.source_dir.resolve()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Reorganizando imágenes"):
        src = (source_dir / row["filename"]).resolve()
        if not src.exists():
            missing_files.append(row["filename"])
            continue

        dest = scene_dir / row["scene_type"] / row["continent"] / src.name
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists():
            duplicates += 1
            if args.on_duplicate == "skip":
                skipped += 1
                continue

        if args.move:
            shutil.move(str(src), str(dest))
        else:
            shutil.copy2(src, dest)
        copied += 1

    print(
        f"\nResumen: copiados/movidos={copied}, omitidos={skipped}, duplicados={duplicates}, faltantes={len(missing_files)}"
    )
    if missing_files:
        print("Algunos archivos no se encontraron en source-dir. Ejemplos:")
        for sample in missing_files[:10]:
            print(f"  - {sample}")

    split_dirs = _generate_splits(
        scene_dir=scene_dir,
        split_base=args.split_base.resolve(),
        test_size=args.split_size,
        seed=args.split_seed,
        clean=args.clean_splits,
    )
    _create_combined_scene_continent_dataset(
        split_dirs=split_dirs,
        output_base=args.split_base.resolve(),
        clean=args.clean_splits,
    )


def _generate_splits(
    scene_dir: Path,
    split_base: Path,
    test_size: float,
    seed: int,
    clean: bool,
) -> List[Tuple[str, Path, Path]]:
    if not scene_dir.exists():
        raise FileNotFoundError(f"No se encontró el directorio de escenas: {scene_dir}")

    split_dirs: List[Tuple[str, Path, Path]] = []
    for scene_path in scene_dir.iterdir():
        if not scene_path.is_dir():
            continue

        scene = scene_path.name
        train_dir = split_base / f"train_{scene}"
        test_dir = split_base / f"test_{scene}"

        if clean:
            for d in (train_dir, test_dir):
                if d.exists():
                    shutil.rmtree(d)
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        for continent_path in scene_path.iterdir():
            if not continent_path.is_dir():
                continue

            images = sorted(continent_path.glob("*.*"))
            if not images:
                continue

            train_imgs, test_imgs = train_test_split(
                images,
                test_size=test_size,
                random_state=seed,
            )

            dest_train = train_dir / continent_path.name
            dest_test = test_dir / continent_path.name
            dest_train.mkdir(parents=True, exist_ok=True)
            dest_test.mkdir(parents=True, exist_ok=True)

            for src in train_imgs:
                shutil.copy2(src, dest_train / src.name)
            for src in test_imgs:
                shutil.copy2(src, dest_test / src.name)

        train_count = sum(len(list((train_dir / c).glob("*.*"))) for c in os.listdir(train_dir))
        test_count = sum(len(list((test_dir / c).glob("*.*"))) for c in os.listdir(test_dir))
        print(
            f"Generados splits para escena '{scene}': {train_count} imágenes en train, {test_count} en test."
        )
        split_dirs.append((scene, train_dir, test_dir))

    return split_dirs


def _create_combined_scene_continent_dataset(
    split_dirs: List[Tuple[str, Path, Path]],
    output_base: Path,
    clean: bool,
) -> None:
    if not split_dirs:
        print("No se generaron splits de escena; se omite dataset combinado.")
        return

    train_combined = output_base / "train_scene_continent"
    test_combined = output_base / "test_scene_continent"

    if clean:
        for directory in (train_combined, test_combined):
            if directory.exists():
                shutil.rmtree(directory)
    train_combined.mkdir(parents=True, exist_ok=True)
    test_combined.mkdir(parents=True, exist_ok=True)

    def _copy_split(src_root: Path, dest_root: Path, scene: str) -> int:
        copied = 0
        for continent_dir in src_root.iterdir():
            if not continent_dir.is_dir():
                continue
            combined_label = f"{scene}-{continent_dir.name}"
            dest_dir = dest_root / combined_label
            dest_dir.mkdir(parents=True, exist_ok=True)
            for img_path in continent_dir.glob("*.*"):
                dest_path = dest_dir / img_path.name
                if dest_path.exists():
                    continue
                shutil.copy2(img_path, dest_path)
                copied += 1
        return copied

    total_train = 0
    total_test = 0
    for scene, train_dir, test_dir in split_dirs:
        total_train += _copy_split(train_dir, train_combined, scene)
        total_test += _copy_split(test_dir, test_combined, scene)

    print(
        f"Dataset combinado generado en {train_combined.parent}: "
        f"{total_train} imágenes en train_scene_continent, {total_test} en test_scene_continent."
    )


def _validate_csv(df: pd.DataFrame) -> None:
    expected = {"filename", "scene_type", "continent", "confidence"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"El CSV no contiene las columnas requeridas: {missing}")


def _normalize_continent(continent: str) -> str:
    return CONTINENT_NORMALIZATION.get(continent, continent)


if __name__ == "__main__":
    main()

