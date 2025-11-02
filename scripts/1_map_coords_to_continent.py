"""
Mapear coordenadas (lat, lon) a continente usando GeoPandas + Cartopy.
Salida: shard0_coords_with_continent.csv (columna extra: continent)
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm
import cartopy.io.shapereader as shpreader


# Entradas y salidas
CSV_IN = 'shard0_coords.csv'
CSV_OUT = 'shard0_coords_with_continent.csv'


# Cargar CSV
df = pd.read_csv(CSV_IN)
print(f'Read {len(df)} rows from {CSV_IN}')

# Validar columnas
assert {'filename', 'lat', 'lon'}.issubset(df.columns), \
    'CSV debe tener columnas filename, lat, lon'


# Cargar shapefile de países desde Cartopy
print("Loading world geometries from Cartopy...")
shpfilename = shpreader.natural_earth(
    resolution='110m',
    category='cultural',
    name='admin_0_countries'
)
world = gpd.read_file(shpfilename)

# Confirmar que tiene la columna 'continent'
if 'CONTINENT' not in world.columns:
    raise ValueError("El shapefile cargado no contiene la columna 'CONTINENT'.")

world = world[['CONTINENT', 'geometry']].rename(columns={'CONTINENT': 'continent'})


# Crear GeoDataFrame con las coordenadas
geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
geo_df = gpd.GeoDataFrame(df.copy(), geometry=geometry, crs=world.crs)

# Spatial join: asignar continente a cada punto
print("Mapping coordinates to continents...")
joined = gpd.sjoin(geo_df, world, how='left', predicate='within')

# Rellenar valores faltantes
joined['continent'] = joined['continent'].fillna('Unknown')

# Guardar CSV final
out = joined.drop(columns=['geometry', 'index_right'])
out.to_csv(CSV_OUT, index=False)
print(f'Saved mapped CSV to {CSV_OUT}')
