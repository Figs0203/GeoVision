"""
Integrar dataset de Google Street View de forma incremental.
Lee coordsSV.csv (lat, lon) y las imágenes de GoogleStreetViewImages/dataset/,
las mapea a continentes y las integra al sistema existente.
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Tuple, List

import cartopy.io.shapereader as shpreader

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
import geopandas as gpd
import pandas as pd
from PIL import Image
from shapely.geometry import Point
from tqdm import tqdm


def _resolve_path(path_str: str) -> Path:
    """Convierte ruta relativa a absoluta."""
    path = Path(path_str)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path.resolve()


def normalize_continent(continent: str) -> str:
    """Normaliza nombres de continentes."""
    if continent in {"North America", "South America"}:
        return "Americas"
    if continent == "Antarctica":
        return "Antarctica"
    return continent


def load_world_geometries() -> gpd.GeoDataFrame:
    """Carga geometrías del mundo desde Cartopy."""
    print("Cargando geometrías del mundo desde Cartopy...")
    shpfilename = shpreader.natural_earth(
        resolution="110m", category="cultural", name="admin_0_countries"
    )
    world = gpd.read_file(shpfilename)
    if "CONTINENT" not in world.columns:
        raise ValueError("El shapefile cargado no contiene la columna 'CONTINENT'.")
    return world[["CONTINENT", "geometry"]].rename(columns={"CONTINENT": "continent"})


def get_starting_index(csv_path: Path) -> int:
    """Obtiene el siguiente índice disponible desde el CSV existente."""
    if not csv_path.exists():
        return 0
    
    try:
        df = pd.read_csv(csv_path)
        if "index" in df.columns:
            max_idx = df["index"].max()
            if pd.isna(max_idx):
                return 0
            return int(max_idx) + 1
        # Si no hay columna index, contar filas
        return len(df)
    except Exception as e:
        print(f"  Advertencia: No se pudo leer índice desde {csv_path}: {e}")
        return 0


def load_streetview_data(streetview_dir: Path) -> Tuple[pd.DataFrame, List[Path]]:
    """Carga CSV de Street View y lista de imágenes ordenadas."""
    # Resolver ruta absoluta para evitar problemas con rutas relativas
    streetview_dir = Path(streetview_dir).resolve()
    
    if not streetview_dir.exists():
        raise FileNotFoundError(
            f"No se encontró el directorio: {streetview_dir}\n"
            f"  Ruta absoluta: {streetview_dir.absolute()}\n"
            f"  Verifica que el directorio existe y que la ruta sea correcta."
        )
    
    # Buscar CSV en múltiples ubicaciones posibles
    csv_path = None
    possible_csv_locations = [
        streetview_dir / "coordsSV.csv",  # En la raíz del directorio
        streetview_dir / "dataset" / "coordsSV.csv",  # Dentro de dataset
    ]
    
    for location in possible_csv_locations:
        if location.exists():
            csv_path = location
            break
    
    if csv_path is None:
        # Buscar CSVs alternativos en el directorio y subdirectorios
        csv_files = list(streetview_dir.glob("**/*.csv"))
        raise FileNotFoundError(
            f"No se encontró coordsSV.csv en las ubicaciones esperadas:\n"
            f"  - {streetview_dir / 'coordsSV.csv'}\n"
            f"  - {streetview_dir / 'dataset' / 'coordsSV.csv'}\n"
            f"  Archivos CSV encontrados: {[f.relative_to(streetview_dir) for f in csv_files]}\n"
            f"  Si el CSV tiene otro nombre o ubicación, muévelo a una de las ubicaciones esperadas."
        )
    
    # Buscar directorio de imágenes
    images_dir = None
    possible_image_locations = [
        streetview_dir / "dataset",  # Dentro de dataset
        streetview_dir,  # En la raíz del directorio
    ]
    
    for location in possible_image_locations:
        if location.exists() and location.is_dir():
            # Verificar que hay imágenes ahí
            image_files = list(location.glob("*.png")) + list(location.glob("*.jpg")) + list(location.glob("*.jpeg"))
            if len(image_files) > 0 or location == streetview_dir / "dataset":
                images_dir = location
                break
    
    if images_dir is None:
        # Buscar directorios con imágenes
        subdirs = [d for d in streetview_dir.iterdir() if d.is_dir()]
        image_dirs = []
        for subdir in subdirs:
            image_files = list(subdir.glob("*.png")) + list(subdir.glob("*.jpg")) + list(subdir.glob("*.jpeg"))
            if len(image_files) > 0:
                image_dirs.append(f"{subdir.name} ({len(image_files)} imágenes)")
        
        raise FileNotFoundError(
            f"No se encontró directorio con imágenes.\n"
            f"  Directorios buscados: {[str(l.relative_to(streetview_dir)) for l in possible_image_locations]}\n"
            f"  Subdirectorios encontrados: {[d.name for d in subdirs]}\n"
            f"  Directorios con imágenes: {image_dirs}\n"
            f"  Si las imágenes están en otro directorio, organiza las imágenes en 'dataset/'."
        )
    
    # Leer CSV
    print(f"Leyendo CSV desde: {csv_path}")
    
    # Intentar leer con encabezados primero
    try:
        df = pd.read_csv(csv_path)
        # Verificar si las columnas son numéricas (probablemente no hay encabezados)
        first_row_values = df.iloc[0].values
        has_headers = False
        
        # Verificar si las columnas tienen nombres típicos de coordenadas
        lat_col = None
        lon_col = None
        for col in df.columns:
            col_lower = str(col).lower()
            if col_lower in ["lat", "latitud", "latitude"]:
                lat_col = col
                has_headers = True
            elif col_lower in ["lon", "lng", "longitud", "longitude", "long"]:
                lon_col = col
                has_headers = True
        
        # Si no encontró encabezados típicos, verificar si las columnas son numéricas
        if not has_headers:
            # Verificar si la primera fila tiene valores numéricos
            if len(df.columns) >= 2:
                try:
                    # Intentar convertir las primeras columnas a float
                    test_val1 = float(str(df.columns[0]))
                    test_val2 = float(str(df.columns[1]))
                    # Si funciona, probablemente no hay encabezados
                    has_headers = False
                    print("  ⚠️  CSV sin encabezados detectado. Usando primera columna como latitud, segunda como longitud.")
                except (ValueError, TypeError):
                    has_headers = True
        
        if not has_headers:
            # Leer CSV sin encabezados
            df = pd.read_csv(csv_path, header=None, names=["lat", "lon"], sep=None, engine='python')
            print(f"  CSV leído sin encabezados: {len(df)} filas")
        else:
            # Normalizar nombres si hay encabezados
            if lat_col is None or lon_col is None:
                # Si hay encabezados pero no son típicos, usar las primeras dos columnas
                if len(df.columns) >= 2:
                    df = df.iloc[:, :2].copy()
                    df.columns = ["lat", "lon"]
                    print(f"  Usando primeras 2 columnas como latitud y longitud")
                else:
                    raise ValueError(
                        f"CSV debe tener al menos 2 columnas (latitud y longitud). "
                        f"Columnas encontradas: {df.columns.tolist()}"
                    )
            else:
                df = df.rename(columns={lat_col: "lat", lon_col: "lon"})
                # Asegurarse de tener solo las columnas lat y lon
                df = df[["lat", "lon"]].copy()
                print(f"  CSV leído con encabezados: {len(df)} filas")
    except Exception as e:
        raise ValueError(f"Error leyendo CSV: {e}")
    
    # Limpiar y convertir valores a float
    # Manejar puntos como separadores de miles (formato europeo) y otros problemas de formato
    def clean_coordinate(value, is_lat=True):
        """Limpia y convierte un valor de coordenada a float."""
        if pd.isna(value):
            return None
        
        # Convertir a string si no lo es
        str_value = str(value).strip()
        
        # Si hay punto y coma, separar (puede ser que ambas coordenadas estén en una celda)
        if ';' in str_value:
            parts = str_value.split(';')
            if len(parts) >= 2:
                # Si es latitud, tomar la primera parte; si es longitud, la segunda
                str_value = parts[0].strip() if is_lat else parts[1].strip()
        
        # Remover espacios
        str_value = str_value.replace(' ', '')
        
        # Reemplazar comas por puntos (formato europeo usa coma como decimal)
        str_value = str_value.replace(',', '.')
        
        # Manejar puntos como separadores de miles
        # Si hay múltiples puntos, necesitamos decidir cuál es el decimal
        if str_value.count('.') > 1:
            parts = str_value.split('.')
            
            # Primero intentar: último punto es decimal (formato europeo)
            if len(parts[-1]) <= 6:  # Hasta 6 dígitos después del punto decimal
                integer_part = ''.join(parts[:-1])
                decimal_part = parts[-1]
                test_value = integer_part + '.' + decimal_part
                try:
                    test_float = float(test_value)
                    # Verificar si el valor es razonable para coordenadas
                    if is_lat and -90 <= test_float <= 90:
                        str_value = test_value
                    elif not is_lat and -180 <= test_float <= 180:
                        str_value = test_value
                    else:
                        # No es razonable, probar remover todos los puntos
                        str_value = str_value.replace('.', '')
                except ValueError:
                    str_value = str_value.replace('.', '')
            else:
                # Todos los puntos son separadores de miles, removerlos todos
                str_value = str_value.replace('.', '')
        
        # También verificar si el valor sin puntos ya es válido
        # (por si los puntos ya fueron procesados correctamente antes)
        elif '.' in str_value:
            try:
                test_float = float(str_value)
                if is_lat and -90 <= test_float <= 90:
                    pass  # Ya es válido, no hacer nada
                elif not is_lat and -180 <= test_float <= 180:
                    pass  # Ya es válido, no hacer nada
                # Si no es válido, continuamos con la normalización más abajo
            except ValueError:
                pass
        
        # Intentar convertir a float
        try:
            result = float(str_value)
            original_result = result
            
            # Primero verificar si el valor ya es válido (puede que ya esté en el formato correcto)
            if is_lat and -90 <= result <= 90:
                # Ya es válido, retornar directamente
                return result
            elif not is_lat and -180 <= result <= 180:
                # Ya es válido, retornar directamente
                return result
            
            # Verificar si el valor parece razonable para coordenadas
            # Si está fuera de rango, intentar normalizar dividiendo por factores comunes
            if is_lat:
                if not (-90 <= result <= 90):
                    # Valor fuera de rango, intentar normalizar
                    if abs(original_result) > 90:
                        # Contar dígitos totales (sin decimales si es entero)
                        try:
                            num_str = str(int(abs(original_result)))
                            num_digits = len(num_str)
                            
                            # Coordenadas típicamente tienen 7-8 dígitos significativos
                            # Si tiene muchos más, probablemente está en nanogrados o similar
                            # Probar múltiples factores desde 1e8 hasta 1e16
                            found_valid = False
                            # Rango amplio de factores posibles (8 a 16 dígitos de diferencia)
                            for factor_power in range(8, 17):
                                factor = 10 ** factor_power
                                test_result = original_result / factor
                                if -90 <= test_result <= 90:
                                    result = test_result
                                    found_valid = True
                                    break
                            
                            # Si no encontró un factor válido, intentar basándose en la cantidad de dígitos
                            if not found_valid and num_digits > 8:
                                excess_digits = num_digits - 7
                                # Probar factores alrededor de la diferencia esperada
                                for offset in [-3, -2, -1, 0, 1, 2, 3]:
                                    factor_power = excess_digits + offset
                                    if 8 <= factor_power <= 16:
                                        factor = 10 ** factor_power
                                        test_result = original_result / factor
                                        if -90 <= test_result <= 90:
                                            result = test_result
                                            break
                        except (ValueError, OverflowError):
                            pass
            
            else:  # longitud
                if not (-180 <= result <= 180):
                    # Valor fuera de rango, intentar normalizar
                    if abs(original_result) > 180:
                        try:
                            num_str = str(int(abs(original_result)))
                            num_digits = len(num_str)
                            
                            # Similar para longitud
                            found_valid = False
                            for factor_power in range(8, 17):
                                factor = 10 ** factor_power
                                test_result = original_result / factor
                                if -180 <= test_result <= 180:
                                    result = test_result
                                    found_valid = True
                                    break
                            
                            if not found_valid and num_digits > 8:
                                excess_digits = num_digits - 7
                                for offset in [-3, -2, -1, 0, 1, 2, 3]:
                                    factor_power = excess_digits + offset
                                    if 8 <= factor_power <= 16:
                                        factor = 10 ** factor_power
                                        test_result = original_result / factor
                                        if -180 <= test_result <= 180:
                                            result = test_result
                                            break
                        except (ValueError, OverflowError):
                            pass
            
            # Verificación final: si todavía está fuera de rango, retornar None
            if is_lat and (result < -90 or result > 90):
                return None
            elif not is_lat and (result < -180 or result > 180):
                return None
                
            return result
        except ValueError:
            print(f"  ⚠️  Advertencia: No se pudo convertir '{value}' a float. Usando NaN.")
            return None
    
    # Aplicar limpieza a ambas columnas
    print("  Limpiando y convirtiendo coordenadas...")
    
    # Mostrar ejemplos de valores originales antes de limpiar
    print(f"  Ejemplos de valores originales (primeras 3 filas):")
    for i in range(min(3, len(df))):
        print(f"    Fila {i}: lat={df.iloc[i]['lat']}, lon={df.iloc[i]['lon']}")
    
    df["lat"] = df["lat"].apply(lambda x: clean_coordinate(x, is_lat=True))
    df["lon"] = df["lon"].apply(lambda x: clean_coordinate(x, is_lat=False))
    
    # Mostrar ejemplos de valores después de limpiar
    print(f"  Ejemplos de valores después de limpiar (primeras 3 filas válidas):")
    valid_df = df.dropna(subset=["lat", "lon"])
    for i in range(min(3, len(valid_df))):
        print(f"    Fila {i}: lat={valid_df.iloc[i]['lat']:.6f}, lon={valid_df.iloc[i]['lon']:.6f}")
    
    # Filtrar filas con valores inválidos
    before_filter = len(df)
    df = df.dropna(subset=["lat", "lon"])
    after_filter = before_filter - len(df)
    if after_filter > 0:
        print(f"  ⚠️  Filas filtradas por coordenadas inválidas: {after_filter}")
    
    # Validar rangos de coordenadas (opcional, pero útil para detectar errores)
    invalid_lat = ((df["lat"] < -90) | (df["lat"] > 90)).sum()
    invalid_lon = ((df["lon"] < -180) | (df["lon"] > 180)).sum()
    if invalid_lat > 0 or invalid_lon > 0:
        print(f"  ⚠️  Advertencia: {invalid_lat + invalid_lon} filas con coordenadas fuera de rango (lat: -90 a 90, lon: -180 a 180)")
        # Filtrar coordenadas inválidas
        df = df[(df["lat"] >= -90) & (df["lat"] <= 90) & (df["lon"] >= -180) & (df["lon"] <= 180)]
    
    print(f"  Coordenadas válidas: {len(df)} filas")
    
    # Obtener lista de imágenes ordenadas
    image_files = sorted(
        [f for f in images_dir.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]],
        key=lambda x: int(x.stem) if x.stem.isdigit() else float("inf")
    )
    
    if len(df) != len(image_files):
        print(f"  ⚠️  Advertencia: CSV tiene {len(df)} filas pero hay {len(image_files)} imágenes")
        # Ajustar al mínimo
        min_len = min(len(df), len(image_files))
        df = df.head(min_len)
        image_files = image_files[:min_len]
        print(f"  Usando solo {min_len} registros/imágenes")
    
    print(f"  CSV: {len(df)} registros")
    print(f"  Imágenes: {len(image_files)} archivos")
    
    return df, image_files


def map_coordinates_to_continents(df: pd.DataFrame, world: gpd.GeoDataFrame) -> pd.Series:
    """Mapea coordenadas a continentes."""
    print("Mapeando coordenadas a continentes...")
    geometry = [Point(xy) for xy in zip(df["lon"], df["lat"])]
    geo_df = gpd.GeoDataFrame(df, geometry=geometry, crs=world.crs)
    joined = gpd.sjoin(geo_df, world, how="left", predicate="within")
    joined["continent"] = joined["continent"].fillna("Unknown")
    joined["continent"] = joined["continent"].map(normalize_continent)
    return joined["continent"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Integrar dataset de Google Street View incrementalmente."
    )
    parser.add_argument(
        "--streetview-dir",
        type=Path,
        default=Path("GoogleStreetViewImages"),
        help="Directorio con coordsSV.csv y carpeta dataset/.",
    )
    parser.add_argument(
        "--csv-coords",
        type=Path,
        default=Path("coords.csv"),
        help="CSV principal con coordenadas (se actualizará).",
    )
    parser.add_argument(
        "--csv-continent",
        type=Path,
        default=Path("coords_with_continent.csv"),
        help="CSV con continentes (se actualizará).",
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=Path("data/images_by_continent"),
        help="Directorio destino para imágenes organizadas por continente.",
    )
    parser.add_argument(
        "--source-images-dir",
        type=Path,
        default=Path("images"),
        help="Directorio temporal donde se copian las imágenes antes de organizarlas.",
    )
    parser.add_argument(
        "--scene-predictions-csv",
        type=Path,
        default=Path("outputs/scene_predictions.csv"),
        help="CSV con clasificaciones de escena (indoor/outdoor). Se actualizará automáticamente si todas las imágenes son outdoor.",
    )
    parser.add_argument(
        "--auto-mark-outdoor",
        action="store_true",
        default=True,
        help="Marcar automáticamente todas las imágenes como outdoor en scene_predictions.csv (default: True).",
    )
    parser.add_argument(
        "--no-auto-mark-outdoor",
        dest="auto_mark_outdoor",
        action="store_false",
        help="No actualizar scene_predictions.csv automáticamente.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Cargar datos de Street View
    sv_df, sv_images = load_streetview_data(args.streetview_dir)
    
    # Obtener índice inicial
    print(f"\nDetectando índice inicial...")
    start_idx = get_starting_index(args.csv_continent)
    print(f"  Índice inicial: {start_idx}")
    
    # Cargar geometrías del mundo
    world = load_world_geometries()
    
    # CRÍTICO: Guardar el índice original del CSV para cada fila
    # Esto asegura que mantengamos la correspondencia CSV <-> Imagen incluso después de filtrar
    sv_df["csv_row_index"] = sv_df.index.tolist()
    
    # Mapear coordenadas a continentes
    sv_df["continent"] = map_coordinates_to_continents(sv_df, world)
    
    # Filtrar Unknown
    unknown_count = (sv_df["continent"] == "Unknown").sum()
    if unknown_count > 0:
        print(f"  ⚠️  Filtrando {unknown_count} imágenes con continente Unknown")
        # Guardar índices válidos ANTES de filtrar (índices originales del CSV)
        valid_mask = sv_df["continent"] != "Unknown"
        valid_csv_indices = sv_df[valid_mask]["csv_row_index"].tolist()
        
        # Filtrar DataFrame
        sv_df = sv_df[valid_mask].copy().reset_index(drop=True)
        
        # Filtrar imágenes correspondientes usando los índices originales del CSV
        # IMPORTANTE: Usamos csv_row_index, no el índice del DataFrame filtrado
        sv_images = [sv_images[i] for i in valid_csv_indices]
        
        print(f"  ✓ Mantenida correspondencia: {len(sv_df)} filas válidas corresponden a {len(sv_images)} imágenes originales")
    
    print(f"\nImágenes válidas a procesar: {len(sv_df)}")
    print("Distribución por continente:")
    print(sv_df["continent"].value_counts())
    
    # Preparar directorios
    args.source_images_dir.mkdir(parents=True, exist_ok=True)
    args.dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Procesar imágenes
    new_records = []
    failed_count = 0
    
    print(f"\nCopiando y organizando imágenes...")
    print(f"  Verificando correspondencia entre CSV e imágenes...")
    print(f"  Total de filas válidas en CSV: {len(sv_df)}")
    print(f"  Total de imágenes correspondientes: {len(sv_images)}")
    
    if len(sv_df) != len(sv_images):
        raise ValueError(
            f"ERROR CRÍTICO: Desincronización detectada!\n"
            f"  Filas válidas en CSV: {len(sv_df)}\n"
            f"  Imágenes correspondientes: {len(sv_images)}\n"
            f"  La correspondencia CSV <-> Imagen está rota. Esto puede causar que las coordenadas no coincidan con las imágenes."
        )
    
    # Mostrar ejemplos de correspondencia antes de procesar (primeras 3)
    if len(sv_df) > 0 and len(sv_images) > 0:
        print(f"\n  ✓ Verificando correspondencia CSV <-> Imagen (primeras 3):")
        for i in range(min(3, len(sv_df), len(sv_images))):
            csv_orig_idx = sv_df.iloc[i].get("csv_row_index", i)
            print(f"    CSV fila original {csv_orig_idx}: lat={sv_df.iloc[i]['lat']:.6f}, lon={sv_df.iloc[i]['lon']:.6f} <-> Imagen: {sv_images[i].name}")
    
    for idx, (row_idx, row) in enumerate(tqdm(sv_df.iterrows(), total=len(sv_df), desc="Procesando")):
        current_index = start_idx + idx
        continent = row["continent"]
        
        # Verificar que tenemos la imagen correspondiente
        if idx >= len(sv_images):
            print(f"\n  ⚠️  Advertencia: No hay imagen correspondiente para la fila {idx} del CSV")
            failed_count += 1
            continue
        
        # Nombre del archivo nuevo (mantener correspondencia 1:1)
        new_filename = f"shard0_{current_index:05d}.jpg"
        
        # Ruta de origen (imagen de Street View) - usar índice para mantener correspondencia
        # CRÍTICO: El orden debe ser CSV fila 0 <-> imagen 0.png, CSV fila 1 <-> imagen 1.png, etc.
        # Las imágenes están ordenadas numéricamente por su nombre (0.png, 1.png, ..., N.png)
        # El CSV está ordenado por fila (primera fila, segunda fila, etc.)
        # Por lo tanto: idx del bucle = índice en CSV = índice en lista de imágenes
        src_image = sv_images[idx]
        
        # Verificar que la imagen existe
        if not src_image.exists():
            print(f"\n  ⚠️  Advertencia: No se encontró la imagen {src_image.name}")
            failed_count += 1
            continue
        
        # Ruta destino temporal (images/)
        temp_dest = args.source_images_dir / new_filename
        
        # Ruta destino final (data/images_by_continent/{continent}/)
        continent_dir = args.dest_dir / continent
        continent_dir.mkdir(exist_ok=True)
        final_dest = continent_dir / new_filename
        
        try:
            # Convertir PNG a JPG si es necesario y copiar
            if src_image.suffix.lower() == ".png":
                img = Image.open(src_image)
                # Convertir RGBA a RGB si es necesario
                if img.mode in ("RGBA", "LA"):
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    if img.mode == "RGBA":
                        background.paste(img, mask=img.split()[3])  # Usar canal alpha
                    else:
                        background.paste(img)
                    img = background
                elif img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(temp_dest, "JPEG", quality=95)
            else:
                shutil.copy2(src_image, temp_dest)
            
            # Copiar a destino final
            shutil.copy2(temp_dest, final_dest)
            
            # Registrar en CSV
            new_records.append({
                "filename": new_filename,
                "lat": row["lat"],
                "lon": row["lon"],
                "index": current_index,
                "continent": continent
            })
        except Exception as e:
            print(f"\n  Error procesando {src_image.name}: {e}")
            failed_count += 1
            continue
    
    if len(new_records) == 0:
        print("\n⚠️  No se procesaron imágenes nuevas.")
        return
    
    print(f"\n{'='*60}")
    print(f"RESUMEN:")
    print(f"  Imágenes procesadas exitosamente: {len(new_records)}")
    print(f"  Imágenes fallidas: {failed_count}")
    
    # Actualizar CSVs
    print(f"\nActualizando CSVs...")
    
    # Actualizar coords_with_continent.csv
    new_df = pd.DataFrame(new_records)
    
    if args.csv_continent.exists():
        existing_df = pd.read_csv(args.csv_continent)
        
        # CRÍTICO: Verificar duplicados por filename antes de combinar
        # Esto evita que se agreguen imágenes que ya fueron procesadas
        existing_filenames = set(existing_df["filename"].tolist()) if "filename" in existing_df.columns else set()
        new_filenames = set(new_df["filename"].tolist())
        
        # Filtrar duplicados
        duplicates = new_filenames & existing_filenames
        if len(duplicates) > 0:
            print(f"\n  ⚠️  Detectadas {len(duplicates)} imágenes ya procesadas (duplicados):")
            for dup in sorted(list(duplicates))[:5]:
                print(f"    - {dup}")
            if len(duplicates) > 5:
                print(f"    ... y {len(duplicates) - 5} más")
            
            # Filtrar duplicados del nuevo DataFrame
            new_df = new_df[~new_df["filename"].isin(duplicates)].copy()
            print(f"  ✓ Filtrando duplicados: {len(new_records) - len(new_df)} imágenes ya existían")
            print(f"  ✓ Se procesarán {len(new_df)} imágenes nuevas")
        
        if len(new_df) > 0:
            # Combinar solo las nuevas
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            # Ordenar por índice
            if "index" in combined_df.columns:
                combined_df = combined_df.sort_values("index").reset_index(drop=True)
        else:
            # No hay imágenes nuevas, usar el existente
            combined_df = existing_df
            print(f"\n  ⚠️  Todas las imágenes ya estaban procesadas. No se agregaron nuevas.")
    else:
        combined_df = new_df
    
    combined_df.to_csv(args.csv_continent, index=False)
    print(f"  ✓ Actualizado {args.csv_continent} con {len(combined_df)} registros totales")
    
    # Actualizar coords.csv
    coords_df = combined_df[["filename", "lat", "lon", "index"]].copy()
    coords_df.to_csv(args.csv_coords, index=False)
    print(f"  ✓ Actualizado {args.csv_coords} con {len(coords_df)} registros totales")
    
    # Actualizar scene_predictions.csv si está habilitado (todas las imágenes son outdoor)
    if args.auto_mark_outdoor and len(new_records) > 0:
        print(f"\nActualizando scene_predictions.csv...")
        scene_predictions_path = _resolve_path(str(args.scene_predictions_csv))
        scene_predictions_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Crear DataFrame con todas las nuevas imágenes marcadas como outdoor
        new_scene_records = []
        for record in new_records:
            new_scene_records.append({
                "filename": record["filename"],
                "scene_type": "outdoor",
                "confidence": 1.0,  # Máxima confianza ya que son de Street View
                "continent": record["continent"],
            })
        
        new_scene_df = pd.DataFrame(new_scene_records)
        
        # Cargar CSV existente si existe
        if scene_predictions_path.exists():
            try:
                existing_scene_df = pd.read_csv(scene_predictions_path)
                # Filtrar registros que ya existen (por si se reejecuta)
                existing_filenames = set(existing_scene_df["filename"])
                new_scene_df = new_scene_df[~new_scene_df["filename"].isin(existing_filenames)]
                
                if len(new_scene_df) > 0:
                    # Combinar
                    combined_scene_df = pd.concat([existing_scene_df, new_scene_df], ignore_index=True)
                    combined_scene_df.to_csv(scene_predictions_path, index=False)
                    print(f"  ✓ Actualizado {scene_predictions_path} con {len(new_scene_df)} nuevas imágenes outdoor")
                    print(f"    Total de registros en scene_predictions.csv: {len(combined_scene_df)}")
                else:
                    print(f"  ✓ Todas las imágenes ya estaban en {scene_predictions_path}")
            except Exception as e:
                print(f"  ⚠️  Advertencia: No se pudo actualizar scene_predictions.csv: {e}")
                # Crear nuevo CSV
                new_scene_df.to_csv(scene_predictions_path, index=False)
                print(f"  ✓ Creado nuevo {scene_predictions_path} con {len(new_scene_df)} imágenes outdoor")
        else:
            # Crear nuevo CSV
            new_scene_df.to_csv(scene_predictions_path, index=False)
            print(f"  ✓ Creado {scene_predictions_path} con {len(new_scene_df)} imágenes outdoor")
    
    print(f"\n{'='*60}")
    print(f"✓ Integración completada exitosamente!")
    print(f"  Total de imágenes en el sistema: {len(combined_df)}")
    print(f"  Nuevas imágenes agregadas: {len(new_records)}")
    if args.auto_mark_outdoor and len(new_records) > 0:
        print(f"  ✓ Todas las nuevas imágenes marcadas como 'outdoor' en scene_predictions.csv")
        print(f"  Nota: Puedes ejecutar el script 2 más tarde si agregas imágenes que no sean outdoor")


if __name__ == "__main__":
    main()

