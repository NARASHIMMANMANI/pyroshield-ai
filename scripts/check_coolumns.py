from pathlib import Path
import pandas as pd

csv_file = Path(
    "datasets/raw/firms/viirs/DL_FIRE_J1V-C2_774643/fire_archive_J1V-C2_774643.csv"
)

df = pd.read_csv(csv_file)

print("=" * 60)
print("Columns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())