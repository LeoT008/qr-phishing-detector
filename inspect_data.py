import pandas as pd

file_path = "data/PhiUSIIL_Phishing_URL_Dataset.csv"

df = pd.read_csv(file_path)

print("Number of rows and columns:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.head())

print("\nLabel counts:")
print(df["label"].value_counts())

print("\nMissing values:")
print(df.isnull().sum().sum())