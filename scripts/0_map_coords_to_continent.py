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

    existing_mapped = {}
    if not args.full_refresh and args.csv_out.exists():
        print(f"Detectado CSV previo: {args.csv_out}")
        df_existing = pd.read_csv(args.csv_out)
        if "continent" in df_existing.columns:
            existing_mapped = dict(zip(df_existing["filename"], df_existing["continent"]))
            print(f"  Filas previamente mapeadas: {len(existing_mapped)}")

    if args.full_refresh:
        print("Modo full-refresh activado: se recalcularán todas las filas.")
        existing_mapped = {}

    world = load_world_geometries()

    df["already_mapped"] = df["filename"].isin(existing_mapped.keys())
    new_rows = df[~df["already_mapped"]].copy()
    old_rows = df[df["already_mapped"]].copy()

    print(f"  Filas ya mapeadas: {len(old_rows)}")
    print(f"  Filas nuevas a mapear: {len(new_rows)}")

    if len(new_rows) > 0:
        geometry = [Point(xy) for xy in zip(new_rows["lon"], new_rows["lat"])]
        geo_df = gpd.GeoDataFrame(new_rows, geometry=geometry, crs=world.crs)
        joined = gpd.sjoin(geo_df, world, how="left", predicate="within")
        joined["continent"] = joined["continent"].fillna("Unknown")
        joined["continent"] = joined["continent"].map(normalize_continent)
        new_rows_mapped = joined.drop(columns=["geometry", "index_right", "already_mapped"])
    else:
        new_rows_mapped = pd.DataFrame()

    if len(old_rows) > 0:
        old_rows["continent"] = old_rows["filename"].map(existing_mapped)
        old_rows = old_rows.drop(columns=["already_mapped"])

    if len(new_rows_mapped) > 0 and len(old_rows) > 0:
        result = pd.concat([old_rows, new_rows_mapped], ignore_index=True)
    elif len(new_rows_mapped) > 0:
        result = new_rows_mapped
    else:
        result = old_rows

    if "index" in result.columns:
        result = result.sort_values("index").reset_index(drop=True)

    print("\nDistribución por continente:")
    print(result["continent"].value_counts())

    result.to_csv(args.csv_out, index=False)
    print(f"\nGuardado CSV final en: {args.csv_out}")
    print(f"Total de registros: {len(result)}")


if __name__ == "__main__":
    main()
