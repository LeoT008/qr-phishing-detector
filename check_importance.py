from pathlib import Path
import pandas as pd
import joblib

folder = Path(__file__).resolve().parent

models = {
    "Original Forest": "random_forest.joblib",
    "Augmented Forest": "random_forest_augmented.joblib",
}

for name, filename in models.items():
    model = joblib.load(folder / "models" / filename)

    # Show which features the forest uses most across its trees.
    importance = pd.Series(
        model.feature_importances_,
        index=model.feature_names_in_
    ).sort_values(ascending=False)

    print(f"\n--- {name} ---")
    print(importance.head(10).round(4).to_string())