"""
Downloader inteligente: Extrae imágenes y filtra por continente.
Se integra automáticamente con shard_0 (continúa índices y CSV).
"""

import msgpack
import os
from tqdm import tqdm
from PIL import Image
import io
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import cartopy.io.shapereader as shpreader


# ===== CONFIGURACIÓN =====
SHARD_FILE = "./shards/shard_3.msg"
SAVE_DIR = "./images"  # ← Directorio destino para las imágenes extraídas
CSV_FILE = "coords.csv"  # ← CSV consolidado de coordenadas

# Filtro: None para extraer todo, o lista de continentes
FILTER_CONTINENTS = ['Asia'] 


# ===== DETECTAR ÍNDICE INICIAL =====
print("Detectando índice inicial desde archivos existentes...")
START_INDEX = 0

if os.path.exists(SAVE_DIR):
    # Buscar el índice más alto de archivos shard0_XXXXX.jpg
    existing_files = [f for f in os.listdir(SAVE_DIR) if f.startswith('shard0_') and f.endswith('.jpg')]
    if existing_files:
        # Extraer números de los nombres: shard0_00042.jpg → 42
        indices = [int(f.replace('shard0_', '').replace('.jpg', '')) for f in existing_files]
        START_INDEX = max(indices) + 1
        print(f"  Encontrados {len(existing_files)} archivos existentes")
        print(f"  Último índice: {max(indices)}")
        print(f"  Nuevo índice inicial: {START_INDEX}")
else:
    print("  No se encontraron archivos existentes, comenzando desde 0")
    os.makedirs(SAVE_DIR, exist_ok=True)


# ===== CARGAR CSV EXISTENTE (SI EXISTE) =====
existing_records = []
if os.path.exists(CSV_FILE):
    print(f"\nCargando CSV existente: {CSV_FILE}")
    existing_df = pd.read_csv(CSV_FILE)
    existing_records = existing_df.to_dict('records')
    print(f"  Registros existentes en CSV: {len(existing_records)}")
else:
    print(f"\nNo se encontró CSV existente, se creará uno nuevo")


# ===== CARGAR GEOMETRÍAS DEL MUNDO =====
print("\nCargando mapa mundial...")
shpfilename = shpreader.natural_earth(
    resolution='110m',
    category='cultural',
    name='admin_0_countries'
)
world = gpd.read_file(shpfilename)
world = world[['CONTINENT', 'geometry']].rename(columns={'CONTINENT': 'continent'})

def normalize_continent(c):
    if c in ['North America', 'South America']:
        return 'Americas'
    if c == 'Antarctica':
        return 'Antarctica'
    return c


def get_continent(lat, lon):
    """Obtiene el continente de una coordenada."""
    try:
        point = Point(lon, lat)
        point_gdf = gpd.GeoDataFrame([{'geometry': point}], crs=world.crs)
        joined = gpd.sjoin(point_gdf, world, how='left', predicate='within')
        
        if len(joined) > 0 and not pd.isna(joined.iloc[0]['continent']):
            continent = joined.iloc[0]['continent']
            return normalize_continent(continent)
        return 'Unknown'
    except:
        return 'Unknown'


# ===== PROCESAMIENTO =====
new_records = []
count = START_INDEX
filtered_count = 0
saved_count = 0

print(f"\nProcesando {SHARD_FILE}...")
if FILTER_CONTINENTS:
    print(f"Filtro activo: Solo extraer {FILTER_CONTINENTS}")
else:
    print("Filtro desactivado: Extraer todas las imágenes")

with open(SHARD_FILE, "rb") as f:
    unpacker = msgpack.Unpacker(f, raw=False)
    
    for record in tqdm(unpacker, desc="Procesando shard_2"):
        img_bytes = record['image']
        lat = record['latitude']
        lon = record['longitude']
        
        # Determinar continente
        continent = get_continent(lat, lon)
        
        # Aplicar filtro (si está activo)
        if FILTER_CONTINENTS and continent not in FILTER_CONTINENTS:
            filtered_count += 1
            continue
        
        # Guardar imagen con mismo formato que shard_0
        fname = f"shard0_{count:05d}.jpg"  # ← MISMO formato: shard0_XXXXX.jpg
        img_path = os.path.join(SAVE_DIR, fname)
        
        try:
            img = Image.open(io.BytesIO(img_bytes))
            img.save(img_path)
            new_records.append({
                'filename': fname,
                'lat': lat,
                'lon': lon,
                'index': count
            })
            count += 1
            saved_count += 1
        except Exception as e:
            print(f"Error {fname}: {e}")
            continue

# ===== COMBINAR CON REGISTROS EXISTENTES =====
print("\nCombinando con datos existentes...")
all_records = existing_records + new_records
combined_df = pd.DataFrame(all_records)

# Guardar CSV combinado (reemplaza el anterior)
combined_df.to_csv(CSV_FILE, index=False)

print(f"\n{'='*60}")
print(f"RESUMEN:")
print(f"  Registros previos en CSV: {len(existing_records)}")
print(f"  Nuevas imágenes guardadas: {saved_count}")
print(f"  Imágenes filtradas (descartadas): {filtered_count}")
print(f"  Total en CSV final: {len(all_records)}")
print(f"\nRango de índices:")
print(f"  Índice inicial: {START_INDEX}")
print(f"  Índice final: {count - 1}")
print(f"\nArchivos actualizados:")
print(f"  - Imágenes: {SAVE_DIR}/ (total: {len(os.listdir(SAVE_DIR))} archivos)")
print(f"  - CSV: {CSV_FILE} (total: {len(all_records)} registros)")
print(f"{'='*60}")