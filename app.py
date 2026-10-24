
from flask import Flask, render_template, request
from eo import create_rgb_image
import json

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None
    aoi_json = ""

    if request.method == "POST":
        aoi_json = request.form.get("aoi", "").strip()

        try:
            if not aoi_json:
                raise ValueError("Please draw an AOI first.")

            aoi = json.loads(aoi_json)

            if aoi.get("type") != "Feature":
                raise ValueError("Invalid GeoJSON feature.")

            geometry = aoi.get("geometry") or {}

            if geometry.get("type") not in ("Polygon", "MultiPolygon"):
                raise ValueError("Please draw a rectangle or polygon.")

            if not geometry.get("coordinates"):
                raise ValueError("The AOI has no coordinates.")

            # Generate the satellite image using the selected geometry.
            result = create_rgb_image(geometry)

        except Exception as exc:
            error = str(exc)
            app.logger.exception("AOI processing failed")

    return render_template(
        "index.html",
        result=result,
        error=error,
        aoi_json=aoi_json
    )


if __name__ == "__main__":
    app.run(debug=True)
