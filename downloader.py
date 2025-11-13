import msgpack
import os
from tqdm import tqdm
from PIL import Image
import io
import pandas as pd

# Configuración de directorios y nombres
SHARD_FILE = "./shards/shard_0.msg"   # Modifica según la ubicación de tu shard
SAVE_DIR = "./images"
CSV_FILE = "coords.csv"

os.makedirs(SAVE_DIR, exist_ok=True)
records = []
count = 0

with open(SHARD_FILE, "rb") as f:
    unpacker = msgpack.Unpacker(f, raw=False)
    for record in tqdm(unpacker, desc="Procesando imágenes"):
        img_bytes = record['image']
        lat = record['latitude']
        lon = record['longitude']
        # Nombre numérico incremental
        fname = f"shard0_{count:05d}.jpg"
        img_path = os.path.join(SAVE_DIR, fname)
        try:
            img = Image.open(io.BytesIO(img_bytes))
            img.save(img_path)
        except Exception as e:
            print(f"Error {fname}: {e}")
            continue
        records.append({'filename': fname, 'lat': lat, 'lon': lon, 'index': count})
        count += 1

# Guardar todo en CSV
pd.DataFrame(records).to_csv(CSV_FILE, index=False)
print(f"¡Listo! {count} imágenes guardadas en {SAVE_DIR}, coordenadas en {CSV_FILE}")
