"""
Clasificar imágenes como indoor/outdoor usando Places365 ResNet-50.
Filtra automáticamente las imágenes 'unknown' y aquellas con confianza < 0.6 (configurable).
Genera CSV con scene_type, confidence y continent.
"""

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from preprocessing.scene_classifier import Places365SceneClassifier


def _resolve_path(path_str: str) -> Path:
    """Convierte ruta relativa a absoluta."""
    path = Path(path_str)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clasificar imágenes como indoor/outdoor usando Places365."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data/images_by_continent"),
        help="Directorio con imágenes organizadas por continente.",
    )
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=Path("coords_with_continent.csv"),
        help="CSV con columnas filename y continent.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/scene_predictions.csv"),
        help="CSV de salida con scene_type y confidence.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.6,
        help="Confianza mínima para aceptar clasificación indoor/outdoor. Si es menor, se marca como 'unknown'.",
    )
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Re-clasificar todas las imágenes, incluso si ya están en el CSV de salida.",
    )
    args = parser.parse_args()

    image_dir = _resolve_path(str(args.image_dir))
    mapping_csv = _resolve_path(str(args.mapping_csv))
    output_path = _resolve_path(str(args.output))

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    if not mapping_csv.exists():
        raise FileNotFoundError(f"Mapping CSV not found: {mapping_csv}")

    # Cargar mapeo de continentes
    print(f"Cargando mapeo de continentes desde {mapping_csv}...")
    coords_df = pd.read_csv(mapping_csv)
    if "filename" not in coords_df.columns or "continent" not in coords_df.columns:
        raise ValueError("CSV debe contener columnas 'filename' y 'continent'.")
    
    # Normalizar nombres de continentes
    continent_norm = {"North America": "Americas", "South America": "Americas"}
    coords_df["continent"] = coords_df["continent"].replace(continent_norm)
    
    filename_to_continent = dict(zip(coords_df["filename"], coords_df["continent"]))

    # Cargar clasificaciones existentes (modo incremental)
    existing_classifications = {}
    if not args.full_refresh and output_path.exists():
        print(f"Detectado CSV previo: {output_path}")
        try:
            existing_df = pd.read_csv(output_path)
            if "filename" in existing_df.columns:
                # Crear diccionario filename -> (scene_type, confidence, continent)
                for _, row in existing_df.iterrows():
                    existing_classifications[row["filename"]] = {
                        "scene_type": row.get("scene_type", "unknown"),
                        "confidence": row.get("confidence", 0.0),
                        "continent": row.get("continent", "Unknown"),
                    }
                print(f"  Clasificaciones existentes cargadas: {len(existing_classifications)}")
        except Exception as e:
            print(f"  Advertencia: No se pudo cargar CSV existente: {e}")
            existing_classifications = {}
    
    if args.full_refresh:
        print("Modo full-refresh activado: se re-clasificarán todas las imágenes.")
        existing_classifications = {}

    # Encontrar todas las imágenes (eliminando duplicados por path absoluto)
    print(f"\nBuscando imágenes en {image_dir}...")
    image_paths_all: List[Path] = []
    
    # Buscar con extensiones normalizadas (Windows no distingue mayúsculas/minúsculas)
    # Usar solo minúsculas para evitar duplicados
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths_all.extend(image_dir.rglob(ext))
    
    if not image_paths_all:
        print("No images found to classify; exiting.")
        return
    
    # Eliminar duplicados por path absoluto (mismo archivo físico)
    seen_paths = set()
    image_paths: List[Path] = []
    duplicates_count = 0
    
    for path in image_paths_all:
        # Usar path absoluto para detectar duplicados reales
        abs_path = str(path.resolve())
        if abs_path not in seen_paths:
            seen_paths.add(abs_path)
            image_paths.append(path)
        else:
            duplicates_count += 1
    
    if duplicates_count > 0:
        print(f"  Advertencia: {duplicates_count} imágenes duplicadas encontradas y omitidas")
    
    print(f"  Total de imágenes únicas encontradas: {len(image_paths)}")
    
    # Separar imágenes ya clasificadas de las nuevas
    images_to_classify: List[Path] = []
    already_classified: List[dict] = []
    
    for path in image_paths:
        filename = path.name
        if filename in existing_classifications:
            # Usar clasificación existente
            existing = existing_classifications[filename]
            already_classified.append({
                "filename": str(path),
                "scene_type": existing["scene_type"],
                "confidence": existing["confidence"],
                "continent": existing["continent"],
            })
        else:
            # Necesita clasificación
            images_to_classify.append(path)
    
    print(f"  Imágenes ya clasificadas: {len(already_classified)}")
    print(f"  Imágenes nuevas a clasificar: {len(images_to_classify)}")
    
    # Clasificar solo imágenes nuevas (si hay)
    if len(images_to_classify) == 0:
        print("\n✓ Todas las imágenes ya están clasificadas. Usa --full-refresh para re-clasificar.")
        new_results = []
    else:
        print(f"\nClasificando {len(images_to_classify)} imágenes nuevas...")
        
        # Cargar clasificador Places365 solo si hay imágenes nuevas
        model_path = ROOT_DIR / "models/places365/resnet50_places365.pth"
        categories_path = ROOT_DIR / "models/places365/categories_places365.txt"
        io_path = ROOT_DIR / "models/places365/IO_places365.txt"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Modelo Places365 no encontrado: {model_path}\n"
                "Descarga desde: http://places2.csail.mit.edu/models_places365/resnet50_places365.pth"
            )
        if not categories_path.exists():
            raise FileNotFoundError(f"Categorías no encontradas: {categories_path}")
        if not io_path.exists():
            raise FileNotFoundError(f"IO split no encontrado: {io_path}")

        print("\nCargando clasificador Places365...")
        classifier = Places365SceneClassifier(
            model_path=model_path,
            categories_path=categories_path,
            io_path=io_path,
            device="cuda" if __import__("torch").cuda.is_available() else "cpu",
            batch_size=32,
        )
        print("Clasificador cargado.")

        # Clasificar solo imágenes nuevas con barra de progreso
        print(f"\n{'='*60}")
        print(f"CLASIFICANDO IMÁGENES NUEVAS")
        print(f"{'='*60}")
        print(f"Total a clasificar: {len(images_to_classify)} imágenes")
        print(f"Tiempo estimado: ~{len(images_to_classify) * 0.05:.1f} minutos (con GPU)")
        print(f"{'='*60}\n")
        
        new_results = classifier.classify_paths(images_to_classify, threshold=0.1, show_progress=True)
        
        print(f"\n{'='*60}")
        print(f"✓ CLASIFICACIÓN COMPLETADA")
        print(f"  Procesadas: {len(new_results)}/{len(images_to_classify)} imágenes")
        print(f"{'='*60}")

    # Combinar resultados existentes con nuevos
    all_results = already_classified + new_results
    
    # Convertir a DataFrame
    results_df = pd.DataFrame(all_results)
    results_df["filename"] = results_df["filename"].apply(lambda p: Path(p).name)

    # Agregar continente a todas las imágenes (nuevas y existentes)
    results_df["continent"] = results_df["filename"].map(filename_to_continent)
    missing_continent = results_df["continent"].isna().sum()
    if missing_continent > 0:
        print(f"\nAdvertencia: {missing_continent} imágenes sin continente asignado")
        # Para imágenes sin continente, intentar obtenerlo del CSV de entrada
        results_df = results_df.dropna(subset=["continent"])

    # Filtrar por confianza mínima: si confidence < min_confidence, marcar como 'unknown'
    # Esto se aplica tanto a imágenes nuevas como existentes (puede cambiar el umbral)
    min_confidence = args.min_confidence
    print(f"\nFiltrando por confianza mínima ({min_confidence})...")
    initial_count = len(results_df)
    low_confidence_mask = results_df["confidence"] < min_confidence
    low_confidence_count = low_confidence_mask.sum()
    
    if low_confidence_count > 0:
        print(f"  Imágenes con confianza < {min_confidence}: {low_confidence_count}")
        # Marcar como 'unknown' las que tienen confianza baja
        results_df.loc[low_confidence_mask, "scene_type"] = "unknown"
        # Mantener confianza original para referencia

    # Filtrar 'unknown' (incluye las originales y las marcadas por confianza baja)
    print("\nFiltrando imágenes 'unknown'...")
    before_unknown_filter = len(results_df)
    results_df = results_df[results_df["scene_type"] != "unknown"].copy()
    filtered_count = before_unknown_filter - len(results_df)
    print(f"  Imágenes filtradas (unknown + baja confianza): {filtered_count}")
    print(f"  Imágenes restantes: {len(results_df)}")

    # Seleccionar columnas finales: filename, scene_type, confidence, continent
    output_df = results_df[["filename", "scene_type", "confidence", "continent"]].copy()

    # Guardar CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"\nGuardado CSV en: {output_path}")
    print(f"Total de registros: {len(output_df)}")
    
    # Mostrar distribución
    print("\nDistribución por escena:")
    print(output_df["scene_type"].value_counts())
    print("\nDistribución por continente:")
    print(output_df["continent"].value_counts())


if __name__ == "__main__":
    main()

