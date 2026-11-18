from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

folder = Path(__file__).resolve().parent
data = pd.read_csv(folder / "data" / "clean_urls.csv")

train, test = train_test_split(
    data,
    test_size=0.20,
    random_state=42,
    stratify=data["label"]
)

output = folder / "data" / "url_splits"
output.mkdir(exist_ok=True)

train.to_csv(output / "train_urls.csv", index=False)
test.to_csv(output / "test_urls.csv", index=False)

print("Training URLs:", len(train))
print("Testing URLs:", len(test))
print("Saved to:", output)