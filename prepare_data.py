from pathlib import Path
import pandas as pd

folder = Path(__file__).resolve().parent

df = pd.read_csv(
    folder / "data" / "PhiUSIIL_Phishing_URL_Dataset.csv"
)

# Keep the URL and its classification label.
data = df[["URL", "label"]].copy()
data["URL"] = data["URL"].str.strip()

# Check whether the same URL has conflicting labels.
conflicts = data.groupby("URL")["label"].nunique()
if (conflicts > 1).any():
    raise ValueError("Conflicting URL labels found. Send me this error.")

# Remove repeated URLs before splitting into training and test data.
before = len(data)
data = data.drop_duplicates(subset="URL")

output = folder / "data" / "clean_urls.csv"
data.to_csv(output, index=False)

print("Duplicate rows removed:", before - len(data))
print("Remaining rows:", len(data))
print("Label counts:")
print(data["label"].value_counts())
print("Saved:", output)