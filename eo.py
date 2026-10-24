from datetime import date, timedelta
from pathlib import Path
import uuid

import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_geom
from PIL import Image, ImageFilter
from pystac_client import Client

STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
MAX_OUTPUT_PIXELS = 2_000_000


def find_satellite_image(geometry):
    """Find a relatively clear Sentinel-2 item intersecting the AOI."""
    client = Client.open(STAC_URL)
    end_date = date.today()
    start_date = end_date - timedelta(days=365)

    search = client.search(
        collections=[COLLECTION],
        intersects=geometry,
        datetime=f"{start_date.isoformat()}/{end_date.isoformat()}",
        query={"eo:cloud_cover": {"lt": 40}},
        max_items=30,
    )
    items = list(search.items())

    if not items:
        search = client.search(
            collections=[COLLECTION],
            intersects=geometry,
            datetime=f"{start_date.isoformat()}/{end_date.isoformat()}",
            max_items=30,
        )
        items = list(search.items())

    if not items:
        raise RuntimeError(
            "No Sentinel-2 scenes were found for this AOI in the last 12 months. "
            "Try a larger AOI or another location."
        )

    items.sort(key=lambda item: item.properties.get("eo:cloud_cover", 100))
    return items[0]


def _read_band(href, geometry):
    """Read one band, clip it to the AOI, and return an 8-bit channel and mask."""
    with rasterio.open(href) as src:
        projected_geometry = transform_geom("EPSG:4326", src.crs, geometry)
        clipped, _ = mask(src, [projected_geometry], crop=True, filled=False)

        band = clipped[0]
        data = np.asarray(band.data, dtype=np.float32)
        valid = ~np.ma.getmaskarray(band) & np.isfinite(data)

    if not np.any(valid):
        raise RuntimeError("The AOI contains no valid pixels in one of the RGB bands.")

    values = data[valid]
    low, high = np.percentile(values, [2, 98])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        channel = np.zeros(data.shape, dtype=np.uint8)
    else:
        scaled = np.clip((data - low) / (high - low), 0, 1)
        channel = (scaled * 255).astype(np.uint8)

    channel[~valid] = 0
    return channel, valid


def create_rgb_image(geometry, output_dir):
    """Create an enhanced RGB PNG clipped to the submitted GeoJSON AOI."""
    item = find_satellite_image(geometry)
    band_hrefs = [item.assets[key].href if key in item.assets else None
                  for key in ("red", "green", "blue")]
    if any(href is None for href in band_hrefs):
        raise RuntimeError("Selected Sentinel-2 scene is missing a red, green, or blue band asset.")

    red, red_valid = _read_band(band_hrefs[0], geometry)
    green, green_valid = _read_band(band_hrefs[1], geometry)
    blue, blue_valid = _read_band(band_hrefs[2], geometry)

    if red.shape != green.shape or red.shape != blue.shape:
        raise RuntimeError("RGB bands have different clipped dimensions. Try a smaller AOI.")

    height, width = red.shape
    if height * width > MAX_OUTPUT_PIXELS:
        raise RuntimeError(
            f"AOI is too large ({width} × {height} pixels). Draw a smaller AOI and try again."
        )

    valid = red_valid & green_valid & blue_valid
    rgb = np.dstack((red, green, blue))
    image = Image.fromarray(rgb, mode="RGB")
    image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=110, threshold=3))

    rgba = image.convert("RGBA")
    rgba.putalpha(Image.fromarray(valid.astype(np.uint8) * 255, mode="L"))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"sentinel_aoi_{uuid.uuid4().hex[:12]}.png"
    rgba.save(output_dir / filename, optimize=True)

    cloud_cover = item.properties.get("eo:cloud_cover")
    return {
        "filename": filename,
        "observation": item.properties.get("datetime", "Unknown"),
        "granule": item.id,
        "cloud_cover": round(float(cloud_cover), 2) if cloud_cover is not None else None,
        "width": width,
        "height": height,
        "valid_pixel_percentage": round(float(valid.mean() * 100), 2),
    }
