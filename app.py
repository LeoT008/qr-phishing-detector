from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, render_template

from decode_qr import decode_image

from features import extract_features

app = Flask(__name__)

# Limit uploaded requests to 5 MB.
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

folder = Path(__file__).resolve().parent

# Load the trained models once when the app starts.
models = {
    "Mixed-source Gradient Boosting": joblib.load(
        folder / "models" / "mixed_source_selected.joblib"
    ),
    "Domain-trained Forest": joblib.load(
        folder / "models" / "random_forest_domain_reproducible.joblib"
    ),
    "Original Forest": joblib.load(
        folder / "models" / "random_forest.joblib"
    ),
    "Augmented Forest": joblib.load(
        folder / "models" / "random_forest_augmented.joblib"
    ),
}




def decode_uploaded_qr(upload):
    """Read and decode an uploaded QR image locally."""

    # Convert uploaded bytes into an OpenCV image.
    image_bytes = np.frombuffer(upload.read(), dtype=np.uint8)

    if image_bytes.size == 0:
        raise ValueError("The uploaded file is empty.")

    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Cannot read this image. Use a PNG or JPEG.")

    # Decode the QR payload without following its link.
    return decode_image(image).strip()


def validate_url(url):
    """Check URL syntax without making a network request."""

    try:
        parsed = urlsplit(url)

        # Accept only HTTP/HTTPS addresses with a hostname.
        if (
            parsed.scheme.lower() not in {"http", "https"}
            or not parsed.hostname
            or any(c.isspace() for c in url)
        ):
            raise ValueError("Invalid URL.")

        # Accessing this property checks the port's validity.
        parsed.port

        # Reject embedded usernames and passwords.
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("Embedded credentials are not supported.")

    except ValueError as exc:
        raise ValueError(
            "Enter a valid HTTP or HTTPS URL without embedded credentials."
        ) from exc


def predict_url(url):
    """Compare predictions from the experimental models."""

    # Reuse the feature extractor used during training.
    features = pd.DataFrame([extract_features(url)])
    results = []

    for name, model in models.items():
        # Match the saved model's expected feature order.
        X = features[list(model.feature_names_in_)]

        # Dataset label 0 means phishing.
        phishing_index = list(model.classes_).index(0)
        score = model.predict_proba(X)[0, phishing_index]
        prediction = int(model.predict(X)[0])

        results.append({
            "name": name,
            "label": (
                "PHISHING" if prediction == 0 else "LEGITIMATE"
            ),
            "score": f"{score:.3f}",
        })

    return results


@app.route("/", methods=["GET", "POST"])
def home():
    url = ""
    checked_url = ""
    results = []
    error = ""

    if request.method == "POST":
        url = request.form.get("url", "").strip()
        upload = request.files.get("qr_image")
        has_image = upload is not None and bool(upload.filename)

        try:
            # Require exactly one input.
            if url and has_image:
                raise ValueError(
                    "Provide a URL or an image, not both."
                )

            if not url and not has_image:
                raise ValueError(
                    "Enter a URL or upload a QR image."
                )

            if has_image:
                checked_url = decode_uploaded_qr(upload)
            else:
                checked_url = url

            # Validate before passing the address to the models.
            validate_url(checked_url)
            results = predict_url(checked_url)

        except ValueError as exc:
            error = str(exc)

        except cv2.error:
            error = "Unable to process this image. Try a clear PNG or JPEG."

    # Automatic escaping displays submitted text safely.
    return render_template(
        "index.html",
        url=url,
        checked_url=checked_url,
        results=results,
        error=error,
    )


@app.errorhandler(413)
def upload_too_large(error):
    # Show a readable message when the request exceeds 5 MB.
    return render_template(
        "index.html",
        url="",
        checked_url="",
        results=[],
        error="Upload too large. Use a smaller image under 5 MB.",
    ), 413


if __name__ == "__main__":
    # Keep the application available only on this computer.
    app.run(host="127.0.0.1", port=5001)
