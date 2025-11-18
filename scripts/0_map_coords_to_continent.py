"""
Mapear coordenadas (lat, lon) a continente usando GeoPandas + Cartopy.
Permite modo incremental para evitar recomputar filas ya mapeadas.
Salida: coords_with_continent.csv (columna extra: continent)
"""

import argparse
from pathlib import Path

import cartopy.io.shapereader as shpreader
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mapear coordenadas a continentes (modo completo o incremental)."
    )
    parser.add_argument(
        "--csv-in",
        type=Path,
        default=Path("coords.csv"),
        help="CSV con columnas filename, lat, lon.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("coords_with_continent.csv"),
        help="CSV de salida con columna continent.",
    )
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Ignorar resultados previos y recalcular todos los registros.",
    )
    return parser.parse_args()


def normalize_continent(continent: str) -> str:
    if continent in {"North America", "South America"}:
        return "Americas"
    if continent == "Antarctica":
        return "Antarctica"
    return continent


def load_world_geometries() -> gpd.GeoDataFrame:
    print("Cargando geometrías del mundo desde Cartopy...")
    shpfilename = shpreader.natural_earth(
        resolution="110m", category="cultural", name="admin_0_countries"
    )
    world = gpd.read_file(shpfilename)
    if "CONTINENT" not in world.columns:
        raise ValueError("El shapefile cargado no contiene la columna 'CONTINENT'.")
    return world[["CONTINENT", "geometry"]].rename(columns={"CONTINENT": "continent"})


def main() -> None:
    args = parse_args()

    if not args.csv_in.exists():
        raise FileNotFoundError(f"No se encontró el CSV de entrada: {args.csv_in}")

    df = pd.read_csv(args.csv_in)
    print(f"Leídas {len(df)} filas desde {args.csv_in}")

    required_cols = {"filename", "lat", "lon"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"El CSV debe contener las columnas {required_cols}")

    # Cargar CSV existente completo (si existe y no es full-refresh)
    df_existing_complete = pd.DataFrame()
    existing_mapped = {}
    if not args.full_refresh and args.csv_out.exists():
        print(f"Detectado CSV previo: {args.csv_out}")
        df_existing_complete = pd.read_csv(args.csv_out)
        if "continent" in df_existing_complete.columns:
            existing_mapped = dict(zip(df_existing_complete["filename"], df_existing_complete["continent"]))
            print(f"  Filas previamente mapeadas: {len(existing_mapped)}")

    if args.full_refresh:
        print("Modo full-refresh activado: se recalcularán todas las filas.")
        existing_mapped = {}
        df_existing_complete = pd.DataFrame()

    world = load_world_geometries()

    # Identificar qué filas del CSV de entrada ya están mapeadas
    df["already_mapped"] = df["filename"].isin(existing_mapped.keys())
    new_rows = df[~df["already_mapped"]].copy()
    updated_rows = df[df["already_mapped"]].copy()

    # Filas del CSV existente que NO están en el CSV de entrada (preservar)
    if len(df_existing_complete) > 0:
        rows_to_preserve = df_existing_complete[~df_existing_complete["filename"].isin(df["filename"])].copy()
        print(f"  Filas del CSV existente a preservar: {len(rows_to_preserve)}")
    else:
        rows_to_preserve = pd.DataFrame()

    print(f"  Filas ya mapeadas (actualizar del CSV entrada): {len(updated_rows)}")
    print(f"  Filas nuevas a mapear: {len(new_rows)}")

    # Mapear nuevas filas
    if len(new_rows) > 0:
        geometry = [Point(xy) for xy in zip(new_rows["lon"], new_rows["lat"])]
        geo_df = gpd.GeoDataFrame(new_rows, geometry=geometry, crs=world.crs)
        joined = gpd.sjoin(geo_df, world, how="left", predicate="within")
        joined["continent"] = joined["continent"].fillna("Unknown")
        joined["continent"] = joined["continent"].map(normalize_continent)
        new_rows_mapped = joined.drop(columns=["geometry", "index_right", "already_mapped"])
    else:
        new_rows_mapped = pd.DataFrame()

    # Actualizar filas que ya estaban mapeadas (usar continente del CSV entrada si cambió)
    if len(updated_rows) > 0:
        # Para las filas actualizadas, recalcular continente (pueden haber cambiado coordenadas)
        geometry = [Point(xy) for xy in zip(updated_rows["lon"], updated_rows["lat"])]
        geo_df = gpd.GeoDataFrame(updated_rows, geometry=geometry, crs=world.crs)
        joined = gpd.sjoin(geo_df, world, how="left", predicate="within")
        joined["continent"] = joined["continent"].fillna("Unknown")
        joined["continent"] = joined["continent"].map(normalize_continent)
        updated_rows_mapped = joined.drop(columns=["geometry", "index_right", "already_mapped"])
    else:
        updated_rows_mapped = pd.DataFrame()

    # Combinar: preservar filas antiguas + actualizadas + nuevas
    parts_to_concat = []
    if len(rows_to_preserve) > 0:
        parts_to_concat.append(rows_to_preserve)
    if len(updated_rows_mapped) > 0:
        parts_to_concat.append(updated_rows_mapped)
    if len(new_rows_mapped) > 0:
        parts_to_concat.append(new_rows_mapped)

    if len(parts_to_concat) > 0:
        result = pd.concat(parts_to_concat, ignore_index=True)
    else:
        result = pd.DataFrame()

    # Ordenar por índice si existe
    if "index" in result.columns:
        result = result.sort_values("index").reset_index(drop=True)

    print("\nDistribución por continente:")
    print(result["continent"].value_counts())

    result.to_csv(args.csv_out, index=False)
    print(f"\nGuardado CSV final en: {args.csv_out}")
    print(f"Total de registros: {len(result)}")
    
    # Sincronizar coords.csv si tiene menos registros que coords_with_continent.csv
    csv_in_path = args.csv_in
    if csv_in_path.exists() and len(result) > len(df):
        print(f"\nSincronizando {csv_in_path}...")
        # Crear coords.csv desde coords_with_continent.csv (sin columna continent)
        cols_to_keep = ["filename", "lat", "lon"]
        if "index" in result.columns:
            cols_to_keep.append("index")
        coords_df = result[cols_to_keep].copy()
        coords_df.to_csv(csv_in_path, index=False)
        print(f"  Actualizado {csv_in_path} con {len(coords_df)} registros")


if __name__ == "__main__":
    main()
