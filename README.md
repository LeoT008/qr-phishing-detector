# QR Guard: QR Code Phishing Detection

QR Guard is an experimental machine-learning project that analyzes URLs entered directly or decoded from QR code images. It extracts features from the address and compares predictions from multiple models **without opening the destination website**.

> **Research prototype:** Predictions can be wrong and do not guarantee that a website is safe. The models showed substantial false-alarm rates on some external data.

## Features

- Paste an HTTP or HTTPS URL or upload a PNG or JPEG QR image.
- Decode QR images locally using ZXing-C++ and OpenCV.
- Analyze 18 URL features, such as length, character counts, and HTTPS usage.
- Compare phishing labels and scores from four experimental models.
- Run the Flask interface locally on your computer.

## How it works

1. The user submits a URL or QR image.
2. The application decodes the image, if needed, and validates the URL.
3. It extracts the same URL features used by the saved models.
4. The models predict whether the URL is phishing or legitimate and display a phishing score.

The application does not inspect page content, follow redirects, or verify that a website is safe. A score is a model output, not a verified real-world probability.

## Requirements

- Python 3.9 was used for development.
- The four saved model files required by `app.py` must be present in `models/`.
- Dependencies are listed in `requirements.txt`.

Saved scikit-learn models can be sensitive to library versions. The project evaluation used scikit-learn 1.6.1, NumPy 2.0.2, pandas 2.3.3, and XGBoost 2.1.4. If a model cannot load in a new environment, check the installed package versions.

## Run the web application

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/LeoT008/qr-phishing-detector.git
cd qr-phishing-detector
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5001** in your browser. Enter one URL or upload one QR image, then select **Analyze destination**. Press `Ctrl+C` in the terminal to stop the server.

On Windows, activate the virtual environment with `.venv\Scripts\activate` instead.

To scan an image in the terminal:

```bash
python scan_qr.py test_qr.png
```

## Models and evaluation

The web app displays results from the original random forest, an augmented random forest, a domain-trained random forest, and mixed-source gradient boosting. Other scripts compare logistic regression, random forest, and XGBoost.

The original random forest achieved **99.57% accuracy** on a 47,074-URL original test split. On a separate 54,982-URL external subset with explicit HTTP/HTTPS schemes, however, it flagged **all legitimate URLs as phishing**. This illustrates that high performance on one dataset does not guarantee generalization.

A follow-up mixed-source model achieved **84.38% accuracy** and a **24.46% legitimate false-alarm rate** on a different, 6,478-URL external pilot. These splits were examined in earlier experiments, so the follow-up is exploratory and should not be treated as an untouched final test. The models use different training procedures and their results are not a controlled comparison of algorithms alone.

See [results/RESULTS_ANALYSIS.md](results/RESULTS_ANALYSIS.md) for the datasets, metrics, limitations, and more comparisons.

## Data and reproducibility

The project used the [UCI PhiUSIIL Phishing URL dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+website+dataset) and the [Kaggle Malicious URLs dataset](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset). The datasets and prepared splits are excluded from this repository. Training and evaluation scripts require those files to be obtained and prepared separately. Some research-comparison models may also need to be retrained if their saved files are not included. Review the dataset terms before redistribution.

## Main project files

- `app.py` — Flask web application.
- `decode_qr.py` and `scan_qr.py` — QR decoding and command-line scanning.
- `features.py` — URL feature extraction.
- `templates/` and `static/` — web interface.
- `train_*.py` — model training experiments.
- `compare_models.py` and `improve_generalization.py` — evaluation and follow-up experiments.
- `results/` — saved evaluation tables and analysis.

## Author

Phuoc Tran — CSCI U599 Senior Seminar, University of South Carolina Upstate.
