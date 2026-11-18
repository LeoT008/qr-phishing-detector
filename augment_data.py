from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
import pandas as pd
from features import extract_features

folder = Path(__file__).resolve().parent
train = pd.read_csv(
    folder / "data" / "url_splits" / "train_urls.csv"
)

# Sample only legitimate training URLs.
sample = train[train["label"] == 1].sample(
    frac=0.5,
    random_state=42
)

extra = []

for url in sample["URL"]:
    parsed = urlsplit(url)

    # Skip addresses that cannot be safely transformed.
    if (
        not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        continue

    # Keep the original hostname and add synthetic paths.
    for path, query in [
        ("/", ""),
        ("/about", ""),
        ("/contact", ""),
        ("/search", "q=hello"),
    ]:
        variant = urlunsplit((
            parsed.scheme,
            parsed.netloc,
            path,
            query,
            ""
        ))
        extra.append({"URL": variant, "label": 1})

    # Experiment with removing the www prefix.
    # This may change the destination, so it is synthetic.
    if parsed.netloc.lower().startswith("www."):
        variant = urlunsplit((
            parsed.scheme,
            parsed.netloc[4:],
            parsed.path,
            parsed.query,
            parsed.fragment
        ))
        extra.append({"URL": variant, "label": 1})

augmented = pd.concat(
    [train, pd.DataFrame(extra)],
    ignore_index=True
)

# Remove generated URLs that also occur in the test set.
test = pd.read_csv(
    folder / "data" / "url_splits" / "test_urls.csv"
)
augmented = augmented[
    ~augmented["URL"].isin(set(test["URL"]))
].copy()

# Remove conflicting labels instead of choosing one.
conflicts = augmented.groupby("URL")["label"].nunique()
conflicting_urls = conflicts[conflicts > 1].index
augmented = augmented[
    ~augmented["URL"].isin(conflicting_urls)
].drop_duplicates(subset="URL")

output = folder / "data" / "augmented"
output.mkdir(exist_ok=True)

augmented.to_csv(output / "train_urls.csv", index=False)

print("Extracting features...", flush=True)
X = pd.DataFrame(
    augmented["URL"].map(extract_features).tolist()
)
y = augmented["label"].reset_index(drop=True)

X.to_csv(output / "X_train.csv", index=False)
y.to_csv(output / "y_train.csv", index=False)

print("Original training rows:", len(train))
print("Augmented training rows:", len(augmented))
print("Conflicting URLs removed:", len(conflicting_urls))
print("Label counts:")
print(y.value_counts())
print("Saved to:", output)