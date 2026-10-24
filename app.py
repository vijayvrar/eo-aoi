import json
import os
from pathlib import Path

from flask import Flask, render_template, request
from eo import create_rgb_image

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "static" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    aoi_json = ""

    if request.method == "POST":
        aoi_json = request.form.get("aoi_geojson", "").strip()
        try:
            if not aoi_json:
                raise ValueError("Please draw a rectangle or polygon on the map first.")
            submitted = json.loads(aoi_json)
            geometry = submitted.get("geometry") if submitted.get("type") == "Feature" else submitted
            if not isinstance(geometry, dict):
                raise ValueError("The AOI is invalid. Please draw it again.")
            if geometry.get("type") not in ("Polygon", "MultiPolygon"):
                raise ValueError("Please draw a polygon or rectangle AOI.")
            result = create_rgb_image(geometry, OUTPUT_DIR)
            result["image_url"] = f"/static/outputs/{result['filename']}"
        except Exception as exc:
            app.logger.exception("AOI processing failed")
            error = str(exc) or "The AOI could not be processed."

    return render_template("index.html", result=result, error=error, aoi_json=aoi_json)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
