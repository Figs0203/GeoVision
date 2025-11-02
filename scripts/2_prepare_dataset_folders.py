# scripts/2_prepare_dataset_folders.py
"""
Crear estructura de carpetas: data/images_by_continent/<continent>/
Opciones de balanceo: undersample, oversample (por duplicado) o dejar para el dataloader.
"""

import os
import shutil
import pandas as pd
from pathlib import Path
from collections import Counter

IMG_SRC_DIR = Path('images_shard0')
CSV_MAPPED = Path('shard0_coords_with_continent.csv')
OUT_DIR = Path('data/images_by_continent')

print("=== Preparando dataset por continente ===")

# Leer CSV mapeado
print("Leyendo archivo CSV con continentes...")
df = pd.read_csv(CSV_MAPPED)
print(f"Total de filas en CSV: {len(df)}")

# Filtrar filas con continente desconocido
print("Filtrando filas con continente 'Unknown'...")
df = df[df['continent'] != 'Unknown'].copy()
print(f"Filas restantes después del filtrado: {len(df)}")

# Normalizar nombres de continentes
print("Normalizando nombres de continentes...")
def normalize_continent(c):
    if c in ['North America', 'South America']:
        return 'Americas'
    if c == 'Antarctica':
        return 'Antarctica'
    return c

df['continent'] = df['continent'].map(normalize_continent)

# Crear carpetas de salida
print("Creando estructura de carpetas por continente...")
OUT_DIR.mkdir(parents=True, exist_ok=True)
for c in df['continent'].unique():
    (OUT_DIR / c).mkdir(parents=True, exist_ok=True)
print("Carpetas creadas correctamente.")

# Copiar imágenes
print("Copiando imágenes a sus carpetas correspondientes (esto puede tardar varios minutos)...")
missing = []
total = len(df)
for i, (_, r) in enumerate(df.iterrows(), start=1):
    src = IMG_SRC_DIR / r['filename']
    dst = OUT_DIR / r['continent'] / r['filename']
    if src.exists():
        shutil.copy2(src, dst)
    else:
        missing.append(str(src))
    if i % 1000 == 0 or i == total:
        print(f"  Progreso: {i}/{total} imágenes procesadas...")

print(f"Copiado finalizado. Imágenes faltantes: {len(missing)}")

# Mostrar conteo de imágenes por continente
print("Calculando distribución final de imágenes por continente...")
counts = {c: len(list((OUT_DIR / c).glob('*.jpg'))) for c in df['continent'].unique()}
for cont, count in counts.items():
    print(f"  {cont}: {count} imágenes")

print("=== Proceso completado exitosamente ===")
