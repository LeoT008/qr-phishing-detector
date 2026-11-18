from pathlib import Path
from urllib.parse import urlsplit
import ipaddress
import pandas as pd


def extract_features(url):
    url = str(url).strip()
    lower = url.lower()

    try:
        parsed = urlsplit(
            url if "://" in url else "//" + url
        )
        host = parsed.hostname or ""
        path = parsed.path
        query = parsed.query
    except ValueError:
        host, path, query = "", "", ""

    try:
        ipaddress.ip_address(host)
        is_ip = 1
    except ValueError:
        is_ip = 0

    length = max(len(url), 1)
    digits = sum(c.isdigit() for c in url)
    letters = sum(c.isalpha() for c in url)

    return {
        "url_length": len(url),
        "host_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "host_dot_count": host.count("."),
        "digit_count": digits,
        "digit_ratio": digits / length,
        "letter_ratio": letters / length,
        "hyphen_count": url.count("-"),
        "at_count": url.count("@"),
        "question_count": url.count("?"),
        "equals_count": url.count("="),
        "ampersand_count": url.count("&"),
        "percent_count": url.count("%"),
        "slash_count": url.count("/"),
        "uses_https": int(lower.startswith("https://")),
        "host_is_ip": is_ip,
        "has_punycode": int("xn--" in host),
    }


if __name__ == "__main__":
    folder = Path(__file__).resolve().parent
    data = pd.read_csv(folder / "data" / "clean_urls.csv")

    print("Extracting features. Please wait...")

    features = pd.DataFrame(
        data["URL"].map(extract_features).tolist()
    )
    features["label"] = data["label"].to_numpy()

    output = folder / "data" / "url_features.csv"
    features.to_csv(output, index=False)

    print("Rows:", len(features))
    print("Number of features:", len(features.columns) - 1)
    print("Missing values:", features.isnull().sum().sum())
    print("Saved:", output)