from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pandas as pd

folder = Path(__file__).resolve().parent
external = folder / "data" / "external"


def comparison_key(url):
    """Ignore protocol when checking overlap; preserve path and query."""
    text = str(url).strip()

    try:
        parsed = urlsplit(
            text if "://" in text else "//" + text
        )

        if (
            not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or any(c.isspace() for c in text)
        ):
            return None

        # Accessing port also checks for invalid port numbers.
        port = parsed.port
        host = parsed.hostname.lower()

        if ":" in host:
            host = "[" + host + "]"

        authority = host
        if port is not None:
            authority += ":" + str(port)

        # Treat an empty path and "/" as the same for overlap checks.
        return urlunsplit((
            "",
            authority,
            parsed.path or "/",
            parsed.query,
            "",  # Fragments are excluded from overlap checks.
        ))
    except ValueError:
        return None


data = pd.read_csv(external / "malicious_phish.csv")

# Keep only the two classes used by this project.
data = data[data["type"].isin(["benign", "phishing"])].copy()
data = data.dropna(subset=["url", "type"])
data["URL"] = data["url"].astype(str).str.strip()

# Match the labels in the original PhiUSIIL dataset.
data["label"] = data["type"].map({
    "benign": 1,
    "phishing": 0,
})

data["key"] = data["URL"].map(comparison_key)
invalid = data["key"].isna().sum()
data = data.dropna(subset=["key"]).copy()

# Remove equivalent addresses with contradictory labels.
label_counts = data.groupby("key")["label"].nunique()
conflicting = set(label_counts[label_counts > 1].index)
data = data[~data["key"].isin(conflicting)].copy()

# Check against the entire original dataset, including its test set.
original = pd.read_csv(folder / "data" / "clean_urls.csv")
original_keys = set(
    original["URL"].map(comparison_key).dropna()
)

overlap = data["key"].isin(original_keys)
overlap_rows = int(overlap.sum())
data = data[~overlap].copy()

# Remove duplicates using the same comparison key.
before = len(data)
data = data.drop_duplicates(subset=["key"]).copy()
duplicates = before - len(data)

# Preserve the original URL text; do not invent a protocol.
data["has_scheme"] = data["URL"].str.match(
    r"(?i)^https?://"
)

output = data[["URL", "label", "has_scheme"]]
output.to_csv(external / "external_urls.csv", index=False)

print("Invalid addresses removed:", invalid)
print("Conflicting address groups removed:", len(conflicting))
print("Rows overlapping original dataset removed:", overlap_rows)
print("Duplicate addresses removed:", duplicates)
print("Remaining external rows:", len(output))

print("\nClass counts: 0 = Phishing, 1 = Legitimate")
print(output["label"].value_counts())

print("\nExplicit HTTP/HTTPS scheme by class:")
print(pd.crosstab(output["label"], output["has_scheme"]))

print("\nSaved:", external / "external_urls.csv")