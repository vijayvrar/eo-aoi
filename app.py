import json
from flask import Flask, render_template, request
from eo import create_rgb_image

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
                raise ValueError("Please draw an Area of Interest (AOI) first.")

            aoi = json.loads(aoi_json)
            geometry = aoi.get("geometry")

            if not geometry:
                raise ValueError("Invalid GeoJSON selected.")

            result = create_rgb_image(geometry)

        except Exception as exc:
            error = str(exc)

    return render_template(
        "index.html", result=result, error=error, aoi_json=aoi_json
    )


if __name__ == "__main__":
    app.run(debug=True)
