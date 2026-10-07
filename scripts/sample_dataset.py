import pandas as pd

INPUT_FILE = "datasets/processed/wildfire_cleaned.csv"
OUTPUT_FILE = "datasets/processed/sample_100.csv"

df = pd.read_csv(INPUT_FILE, nrows=100)

df.to_csv(OUTPUT_FILE, index=False)

print("Sample created successfully.")
print(df.head())