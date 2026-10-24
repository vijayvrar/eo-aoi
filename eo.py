
import os
import uuid
from datetime import date

import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_geom
from pystac_client import Client
from PIL import Image, ImageFilter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "static", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initial safety limit for the generated image.
# This is a post-read check, not a complete memory guard.
MAX_OUTPUT_PIXELS = 2_000_000


def find_satellite_image(geometry):
    catalog = Client.open(
        "https://earth-search.aws.element84.com/v1"
    )

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=geometry,
        datetime=f"2026-01-01/{date.today().isoformat()}",
        query={"eo:cloud_cover": {"lt": 40}},
        max_items=20,
    )

    items = list(search.items())

    if not items:
        raise ValueError(
            "No Sentinel-2 scenes found for this AOI. "
            "Try another location or date range."
        )

    items.sort(
        key=lambda item: item.properties.get(
            "eo:cloud_cover", 100
        )
    )
    return items[0]


def read_clipped_band(asset_url, geometry):
    with rasterio.open(asset_url) as src:
        clipped_geometry = transform_geom(
            "EPSG:4326", src.crs, geometry
        )

        data, _ = mask(
            src,
            [clipped_geometry],
            crop=True,
            filled=False,
        )

        band = np.ma.masked_invalid(
            data[0].astype(np.float32)
        )

        # Copy only the 2D mask and release the masked
        # raster before returning the band's data.
        invalid = np.ma.getmaskarray(band).copy()
        values = band.data

        valid = ~invalid
        valid &= np.isfinite(values)

        if not valid.any():
            raise ValueError(
                "The AOI contains no valid pixels in an RGB band."
            )

        # Calculate display stretch using valid pixels only.
        low, high = np.percentile(
            values[valid], [2, 98]
        )

        if high <= low:
            raise ValueError(
                "Unable to enhance one of the RGB bands."
            )

        # Create a compact 8-bit channel instead of retaining
        # three full-size floating-point bands simultaneously.
        channel = np.clip(
            (values - low) / (high - low), 0, 1
        )
        channel = np.power(channel, 0.8)
        channel = (channel * 255).astype(np.uint8)

        del band, data, values, invalid

        return channel, valid


def create_rgb_image(geometry):
    item = find_satellite_image(geometry)

    for name in ("red", "green", "blue"):
        if name not in item.assets:
            raise ValueError(
                f"Sentinel-2 scene is missing the {name} band."
            )

    channels = []
    common_valid = None
    expected_shape = None

    for name in ("red", "green", "blue"):
        channel, band_valid = read_clipped_band(
            item.assets[name].href, geometry
        )

        if expected_shape is None:
            expected_shape = channel.shape
        elif channel.shape != expected_shape:
            raise ValueError(
                "RGB band dimensions differ. "
                "Try another scene or AOI."
            )

        # Combine validity masks without stacking the float bands.
        if common_valid is None:
            common_valid = band_valid.copy()
        else:
            common_valid &= band_valid

        channels.append(channel)
        del channel, band_valid

    height, width = expected_shape
    pixel_count = height * width

    if pixel_count > MAX_OUTPUT_PIXELS:
        raise ValueError(
            f"This AOI produces an image of {width} x {height} "
            f"pixels, which exceeds the current limit. "
            "Please draw a smaller AOI."
        )

    if not common_valid.any():
        raise ValueError(
            "The selected AOI contains no valid RGB pixels."
        )

    # Assemble the final 8-bit RGB output.
    rgb = np.stack(channels, axis=-1)
    del channels

    # Sharpen RGB, keeping AOI transparency separate.
    image = Image.fromarray(rgb)
    del rgb

    image = image.filter(
        ImageFilter.UnsharpMask(
            radius=1, percent=80, threshold=3
        )
    )

    # Apply the common mask so pixels outside the AOI
    # remain transparent.
    alpha = np.where(
        common_valid, 255, 0
    ).astype(np.uint8)

    image.putalpha(Image.fromarray(alpha))
    del alpha, common_valid

    filename = f"sentinel_aoi_{uuid.uuid4().hex[:12]}.png"
    output_path = os.path.join(OUTPUT_DIR, filename)
    image.save(output_path)

    width, height = image.size
    valid_percentage = round(
        float(np.count_nonzero(
            np.asarray(image.getchannel("A"))
        ) / (width * height) * 100),
        2,
    )

    result = {
        "filename": filename,
        "observation": str(
            item.properties.get("datetime", "Unknown")
        ),
        "granule": item.id,
        "cloud_cover": item.properties.get("eo:cloud_cover"),
        "width": width,
        "height": height,
        "valid_pixel_percentage": valid_percentage,
    }

    print(
        f"AOI image generated: {width}x{height}; "
        f"valid pixels: {valid_percentage}%; scene: {item.id}"
    )

    image.close()
    return result
