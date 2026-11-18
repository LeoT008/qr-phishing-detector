from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

folder = Path(__file__).resolve().parent
data = pd.read_csv(folder / "data" / "url_features.csv")

X = data.drop(columns=["label"])
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

output = folder / "data" / "splits"
output.mkdir(exist_ok=True)

X_train.to_csv(output / "X_train.csv", index=False)
X_test.to_csv(output / "X_test.csv", index=False)
y_train.to_csv(output / "y_train.csv", index=False)
y_test.to_csv(output / "y_test.csv", index=False)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Training label counts:")
print(y_train.value_counts())
print("Testing label counts:")
print(y_test.value_counts())
print("Saved to:", output)