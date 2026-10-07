import pandas as pd
from pathlib import Path

INPUT_FILE = "datasets/final/ml_dataset.csv"

print("=" * 60)
print("Exploratory Data Analysis")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print("\nDataset Shape")
print(df.shape)

print("\nData Types")
print(df.dtypes)

print("\nClass Distribution")
print(df["fire"].value_counts())

print("\nPercentage Distribution")
print(df["fire"].value_counts(normalize=True) * 100)

print("\nStatistical Summary")
print(df.describe())

corr = df.corr(numeric_only=True)

Path("datasets/final").mkdir(parents=True, exist_ok=True)
corr.to_csv("datasets/final/correlation_matrix.csv")

print("\nCorrelation matrix saved.")