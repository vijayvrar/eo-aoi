import os
import uuid
import gc
from datetime import date
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_geom
from pystac_client import Client
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "static", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Strict limit to prevent Render 512MB RAM crashes
MAX_PIXELS = 1_000_000 

def create_rgb_image(geometry):
    client = Client.open("https://earth-search.aws.element84.com/v1")
    search = client.search(
        collections=["sentinel-2-l2a"],
        intersects=geometry,
        datetime=f"2026-01-01/{date.today().isoformat()}",
        query={"eo:cloud_cover": {"lt": 35}},
        max_items=10,
    )

    items = list(search.items())
    if not items:
        raise ValueError("No cloud-free Sentinel-2 scenes found for this AOI.")

    item = items[0]
    channels = []

    for band_name in ("red", "green", "blue"):
        href = item.assets[band_name].href
        with rasterio.open(href) as src:
            clipped_geom = transform_geom("EPSG:4326", src.crs, geometry)
            
            # Mask and crop with downsampling safeguard if needed
            data, _ = mask(src, [clipped_geom], crop=True, filled=True)
            band = data[0].astype(np.float32)

            # Check pixel dimensions to avoid RAM spike
            if band.size > MAX_PIXELS:
                raise ValueError("Selected area is too large for the free server tier. Please draw a smaller polygon.")

            valid_pixels = band[band > 0]
            if valid_pixels.size > 0:
                p_low, p_high = np.percentile(valid_pixels, [2, 98])
            else:
                p_low, p_high = 0, 1

            if p_high <= p_low:
                p_high = p_low + 1

            scaled = np.clip((band - p_low) / (p_high - p_low), 0, 1)
            channels.append((scaled * 255).astype(np.uint8))
            
            # Free memory immediately
            del band, data, valid_pixels

    rgb = np.stack(channels, axis=-1)
    del channels
    
    img = Image.fromarray(rgb)
    del rgb

    filename = f"sentinel_{uuid.uuid4().hex[:8]}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    img.save(filepath)
    img.close()

    # Force Python garbage collection to clear RAM for Render
    gc.collect()

    return {
        "filename": filename,
        "observation": item.properties.get("datetime", "Unknown")[:10],
        "granule": item.id,
        "cloud_cover": item.properties.get("eo:cloud_cover"),
    }
