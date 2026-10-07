from pathlib import Path
import pandas as pd

# Change this path if necessary
csv_file = Path(
    "datasets/raw/firms/modis/DL_FIRE_M-C61_774642/fire_archive_M-C61_774642.csv"
)

df = pd.read_csv(csv_file)

print("=" * 60)
print("Columns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())