from pathlib import Path
import pandas as pd

folder = Path(__file__).resolve().parent

# Inspect only the original training data.
data = pd.read_csv(
    folder / "data" / "url_splits" / "train_urls.csv"
)

data["class"] = data["label"].map({
    0: "Phishing",
    1: "Legitimate"
})

# Check the URL text without visiting any website.
data["uses_https"] = (
    data["URL"].str.lower().str.startswith("https://")
)

print("Training URL counts:")
print(pd.crosstab(data["class"], data["uses_https"]))

print("\nPercentage using HTTPS:")
percentages = (
    data.groupby("class")["uses_https"].mean() * 100
)
print(percentages.round(2).to_string())