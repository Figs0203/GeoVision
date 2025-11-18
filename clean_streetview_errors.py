"""
Script para limpiar imágenes procesadas incorrectamente por integrate_streetview.py

Este script elimina las imágenes que fueron procesadas antes de la corrección
del bug de correspondencia CSV <-> Imagen. Identifica imágenes procesadas desde
un índice específico y las elimina de los CSVs y directorios.
"""

import argparse
import pandas as pd
from pathlib import Path
from typing import Tuple, Set


def identify_streetview_images(
    csv_path: Path,
    streetview_start_index: int,
) -> Tuple[pd.DataFrame, Set[str]]:
    """
    Identifica imágenes procesadas desde Street View que pueden estar incorrectas.
    
    Args:
        csv_path: Ruta al CSV con continentes
        streetview_start_index: Índice desde el cual empezaron a procesarse imágenes de Street View
        
    Returns:
        DataFrame con las imágenes identificadas y set con los filenames
    """
    if not csv_path.exists():
        print(f"⚠️  No se encontró {csv_path}")
        return pd.DataFrame(), set()
    
    df = pd.read_csv(csv_path)
    
    if "index" not in df.columns:
        print("⚠️  El CSV no tiene columna 'index'")
        return pd.DataFrame(), set()
    
    # Filtrar imágenes con índice >= streetview_start_index
    sv_images = df[df["index"] >= streetview_start_index].copy()
    
    return sv_images, set(sv_images["filename"].tolist())


def clean_images(
    filenames: Set[str],
    images_dir: Path,
    continent_dir: Path,
    scene_predictions_csv: Path,
) -> None:
    """
    Elimina imágenes de los directorios y CSVs.
    
    Args:
        filenames: Set de nombres de archivos a eliminar
        images_dir: Directorio temporal de imágenes
        continent_dir: Directorio de imágenes por continente
        scene_predictions_csv: CSV con clasificaciones de escena
    """
    deleted_images = 0
    deleted_continent = 0
    
    print(f"\nEliminando imágenes de directorios...")
    
    # Eliminar de images/
    if images_dir.exists():
        for filename in filenames:
            img_path = images_dir / filename
            if img_path.exists():
                img_path.unlink()
                deleted_images += 1
    
    # Eliminar de data/images_by_continent/{continent}/
    if continent_dir.exists():
        for continent_subdir in continent_dir.iterdir():
            if continent_subdir.is_dir():
                for filename in filenames:
                    img_path = continent_subdir / filename
                    if img_path.exists():
                        img_path.unlink()
                        deleted_continent += 1
    
    print(f"  ✓ Eliminadas {deleted_images} imágenes de {images_dir}")
    print(f"  ✓ Eliminadas {deleted_continent} imágenes de {continent_dir}")
    
    # Eliminar de scene_predictions.csv
    if scene_predictions_csv.exists():
        try:
            df = pd.read_csv(scene_predictions_csv)
            before = len(df)
            df = df[~df["filename"].isin(filenames)]
            after = len(df)
            
            if before > after:
                df.to_csv(scene_predictions_csv, index=False)
                print(f"  ✓ Eliminados {before - after} registros de {scene_predictions_csv}")
        except Exception as e:
            print(f"  ⚠️  Error actualizando {scene_predictions_csv}: {e}")


def clean_csvs(
    filenames: Set[str],
    csv_coords: Path,
    csv_continent: Path,
) -> None:
    """
    Elimina registros de los CSVs.
    
    Args:
        filenames: Set de nombres de archivos a eliminar
        csv_coords: CSV principal con coordenadas
        csv_continent: CSV con continentes
    """
    print(f"\nEliminando registros de CSVs...")
    
    # Limpiar coords_with_continent.csv
    if csv_continent.exists():
        try:
            df = pd.read_csv(csv_continent)
            before = len(df)
            df = df[~df["filename"].isin(filenames)]
            after = len(df)
            
            if before > after:
                df.to_csv(csv_continent, index=False)
                print(f"  ✓ Eliminados {before - after} registros de {csv_continent}")
                print(f"    Registros restantes: {after}")
        except Exception as e:
            print(f"  ⚠️  Error actualizando {csv_continent}: {e}")
    
    # Limpiar coords.csv
    if csv_coords.exists():
        try:
            df = pd.read_csv(csv_coords)
            before = len(df)
            df = df[~df["filename"].isin(filenames)]
            after = len(df)
            
            if before > after:
                df.to_csv(csv_coords, index=False)
                print(f"  ✓ Eliminados {before - after} registros de {csv_coords}")
                print(f"    Registros restantes: {after}")
        except Exception as e:
            print(f"  ⚠️  Error actualizando {csv_coords}: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Limpiar imágenes procesadas incorrectamente por integrate_streetview.py"
    )
    parser.add_argument(
        "--streetview-start-index",
        type=int,
        required=True,
        help="Índice desde el cual empezaron a procesarse imágenes de Street View (ej: 41036)",
    )
    parser.add_argument(
        "--csv-continent",
        type=Path,
        default=Path("coords_with_continent.csv"),
        help="CSV con continentes",
    )
    parser.add_argument(
        "--csv-coords",
        type=Path,
        default=Path("coords.csv"),
        help="CSV principal con coordenadas",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("images"),
        help="Directorio temporal de imágenes",
    )
    parser.add_argument(
        "--continent-dir",
        type=Path,
        default=Path("data/images_by_continent"),
        help="Directorio de imágenes por continente",
    )
    parser.add_argument(
        "--scene-predictions-csv",
        type=Path,
        default=Path("outputs/scene_predictions.csv"),
        help="CSV con clasificaciones de escena",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo mostrar qué se eliminaría, sin hacer cambios",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    print("="*60)
    print("LIMPIADOR DE IMÁGENES STREET VIEW INCORRECTAS")
    print("="*60)
    print(f"\nÍndice de inicio Street View: {args.streetview_start_index}")
    
    if args.dry_run:
        print("\n⚠️  MODO DRY-RUN: Solo se mostrará lo que se eliminaría")
    
    # Identificar imágenes a eliminar
    print(f"\nIdentificando imágenes de Street View...")
    sv_df, sv_filenames = identify_streetview_images(
        args.csv_continent,
        args.streetview_start_index
    )
    
    if len(sv_filenames) == 0:
        print("  ✓ No se encontraron imágenes de Street View para eliminar")
        return
    
    print(f"\n  ⚠️  Se encontraron {len(sv_filenames)} imágenes de Street View:")
    print(f"    Índices: {sv_df['index'].min()} - {sv_df['index'].max()}")
    print(f"    Continentes: {sv_df['continent'].value_counts().to_dict()}")
    
    # Mostrar ejemplos
    print(f"\n  Ejemplos de imágenes a eliminar (primeras 5):")
    for idx, row in sv_df.head(5).iterrows():
        print(f"    - {row['filename']} (índice {row['index']}, {row['continent']})")
    
    if args.dry_run:
        print(f"\n⚠️  DRY-RUN: Se eliminarían {len(sv_filenames)} imágenes")
        print(f"  - De {args.images_dir}: ~{len(sv_filenames)} imágenes")
        print(f"  - De {args.continent_dir}: ~{len(sv_filenames)} imágenes")
        print(f"  - De {args.csv_continent}: {len(sv_filenames)} registros")
        print(f"  - De {args.csv_coords}: {len(sv_filenames)} registros")
        return
    
    # Confirmar
    print(f"\n⚠️  ATENCIÓN: Se eliminarán {len(sv_filenames)} imágenes y sus registros de los CSVs")
    response = input("  ¿Continuar? (escriba 'SI' para confirmar): ")
    if response != "SI":
        print("  ✗ Operación cancelada")
        return
    
    # Limpiar
    clean_images(
        sv_filenames,
        args.images_dir,
        args.continent_dir,
        args.scene_predictions_csv,
    )
    
    clean_csvs(
        sv_filenames,
        args.csv_coords,
        args.csv_continent,
    )
    
    print(f"\n{'='*60}")
    print(f"✓ Limpieza completada!")
    print(f"  Total de imágenes eliminadas: {len(sv_filenames)}")
    print(f"\n  Ahora puedes reejecutar integrate_streetview.py para procesar")
    print(f"  las imágenes correctamente con la correspondencia CSV <-> Imagen corregida.")


if __name__ == "__main__":
    main()

