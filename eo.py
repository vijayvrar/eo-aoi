
import os
from datetime import date

import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_geom
from pystac_client import Client
from PIL import Image, ImageFilter


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "static",
    "outputs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# 1. Search Sentinel-2 imagery
# --------------------------------------------------

def find_satellite_image(geometry):

    catalog = Client.open(
        "https://earth-search.aws.element84.com/v1"
    )

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=geometry,
        datetime=f"2026-01-01/{date.today().isoformat()}",
        query={
            "eo:cloud_cover": {
                "lt": 40
            }
        },
        max_items=20
    )

    items = list(search.items())

    if not items:
        raise ValueError(
            "No Sentinel-2 scenes found for this AOI. "
            "Try a different location or date range."
        )

    # Select the least-cloudy scene among returned candidates.
    items.sort(
        key=lambda item: item.properties.get(
            "eo:cloud_cover", 100
        )
    )

    return items[0]


# --------------------------------------------------
# 2. Read a band and preserve its raster mask
# --------------------------------------------------

def read_clipped_band(asset_url, geometry):

    with rasterio.open(asset_url) as src:

        # AOI coordinates from Leaflet are longitude/latitude
        # in EPSG:4326. Convert to the raster's CRS.
        clipped_geometry = transform_geom(
            "EPSG:4326",
            src.crs,
            geometry
        )

        # crop=True limits the output to the polygon's
        # bounding rectangle. filled=False preserves
        # the mask outside the actual polygon.
        data, _ = mask(
            src,
            [clipped_geometry],
            crop=True,
            filled=False
        )

        band = np.ma.masked_invalid(
            data[0].astype(np.float32)
        )

        return band


# --------------------------------------------------
# 3. Create an RGB image with polygon transparency
# --------------------------------------------------

def create_rgb_image(geometry):

    item = find_satellite_image(geometry)

    required_bands = [
        "red",
        "green",
        "blue"
    ]

    for band_name in required_bands:

        if band_name not in item.assets:
            raise ValueError(
                f"Sentinel-2 scene is missing "
                f"the {band_name} band."
            )

    # Read all three bands using the same AOI.
    red = read_clipped_band(
        item.assets["red"].href,
        geometry
    )

    green = read_clipped_band(
        item.assets["green"].href,
        geometry
    )

    blue = read_clipped_band(
        item.assets["blue"].href,
        geometry
    )

    # Ensure all bands have matching dimensions.
    if (
        red.shape != green.shape
        or red.shape != blue.shape
    ):
        raise ValueError(
            "RGB band dimensions differ. "
            "Cannot safely combine these bands."
        )

    # Stack the masked arrays into H x W x 3.
    rgb = np.ma.stack(
        [red, green, blue],
        axis=-1
    )

    # A pixel is valid only if all three bands are valid.
    # Do not automatically discard pixels just because
    # one or more band values are zero.
    band_masks = np.ma.getmaskarray(rgb)

    valid = ~band_masks.any(axis=2)

    # Additional protection against invalid numeric values.
    valid &= np.isfinite(rgb.data).all(axis=2)

    if not valid.any():
        raise ValueError(
            "The selected AOI contains no valid RGB pixels."
        )

    # --------------------------------------------------
    # 4. Enhance each RGB channel independently
    # --------------------------------------------------

    source_pixels = rgb.data
    enhanced_channels = []

    for channel in range(3):

        channel_data = source_pixels[:, :, channel]

        valid_values = channel_data[valid]

        low, high = np.percentile(
            valid_values,
            [2, 98]
        )

        if high <= low:
            raise ValueError(
                "Unable to enhance one of the RGB bands."
            )

        normalized = np.clip(
            (channel_data - low) / (high - low),
            0,
            1
        )

        # Mild gamma correction for display.
        normalized = np.power(
            normalized,
            0.8
        )

        enhanced_channels.append(
            (normalized * 255).astype(np.uint8)
        )

    rgb_pixels = np.stack(
        enhanced_channels,
        axis=-1
    )

    # Sharpen RGB only. Build transparency separately
    # so image filtering cannot alter the AOI mask.
    rgb_image = Image.fromarray(rgb_pixels)

    rgb_image = rgb_image.filter(
        ImageFilter.UnsharpMask(
            radius=1,
            percent=80,
            threshold=3
        )
    )

    rgb_pixels = np.array(rgb_image)

    # --------------------------------------------------
    # 5. Create alpha channel from the raster mask
    # --------------------------------------------------

    alpha = np.where(
        valid,
        255,
        0
    ).astype(np.uint8)

    rgba = np.dstack(
        [
            rgb_pixels,
            alpha
        ]
    )

    output_image = Image.fromarray(rgba)

    # --------------------------------------------------
    # 6. Save the output image
    # --------------------------------------------------

    filename = "sentinel_aoi.png"

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    output_image.save(output_path)

    # Calculate the proportion of valid pixels.
    valid_percentage = round(
        float(valid.mean() * 100),
        2
    )

    print("AOI image saved:", output_path)
    print("Image dimensions:", output_image.size)
    print("Valid pixel percentage:", valid_percentage)
    print("Scene ID:", item.id)

    return {
        "filename": filename,
        "observation": str(
            item.properties.get(
                "datetime",
                "Unknown"
            )
        ),
        "granule": item.id,
        "cloud_cover": item.properties.get(
            "eo:cloud_cover"
        ),
        "width": output_image.width,
        "height": output_image.height,
        "valid_pixel_percentage": valid_percentage
    }
