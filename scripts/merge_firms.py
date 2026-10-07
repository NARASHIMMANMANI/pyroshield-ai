from pathlib import Path
import pandas as pd

ROOT = Path("datasets/raw/firms")
OUTPUT = Path("datasets/processed/wildfire_events_2020_2024.csv")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

first_file = True
total_rows = 0
file_count = 0

for source in ["modis", "viirs"]:

    folder = ROOT / source

    for csv_file in folder.rglob("*.csv"):

        print(f"Reading {csv_file.name}")

        df = pd.read_csv(csv_file)

        df["source"] = source.upper()

        total_rows += len(df)
        file_count += 1

        df.to_csv(
            OUTPUT,
            mode="w" if first_file else "a",
            header=first_file,
            index=False
        )

        first_file = False

        print(f"Rows in file: {len(df):,}")

print("\n" + "=" * 60)
print("Merge Completed Successfully")
print("=" * 60)
print(f"Files merged : {file_count}")
print(f"Total rows   : {total_rows:,}")
print(f"Saved to     : {OUTPUT}")