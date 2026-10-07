from pathlib import Path
import pandas as pd

# ======================================================
# File Paths
# ======================================================

INPUT_FILE = "datasets/processed/wildfire_events_2020_2024.csv"
OUTPUT_FILE = "datasets/processed/wildfire_cleaned.csv"

CHUNK_SIZE = 100000

Path("datasets/processed").mkdir(parents=True, exist_ok=True)

first_chunk = True

total_rows = 0
clean_rows = 0

print("=" * 60)
print("Cleaning Wildfire Dataset")
print("=" * 60)

# ======================================================
# Process in Chunks
# ======================================================

for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE):

    total_rows += len(chunk)

    # Remove rows with missing essential values
    chunk = chunk.dropna(subset=[
        "latitude",
        "longitude",
        "acq_date"
    ])

    # Convert latitude/longitude to numeric
    chunk["latitude"] = pd.to_numeric(chunk["latitude"], errors="coerce")
    chunk["longitude"] = pd.to_numeric(chunk["longitude"], errors="coerce")

    chunk = chunk.dropna(subset=["latitude", "longitude"])

    # Remove invalid coordinates
    chunk = chunk[
        (chunk["latitude"] >= -90) &
        (chunk["latitude"] <= 90) &
        (chunk["longitude"] >= -180) &
        (chunk["longitude"] <= 180)
    ]

    # Convert date
    chunk["acq_date"] = pd.to_datetime(
        chunk["acq_date"],
        errors="coerce"
    )

    chunk = chunk.dropna(subset=["acq_date"])

    # Remove duplicate wildfire events
    chunk = chunk.drop_duplicates(
        subset=["latitude", "longitude", "acq_date"]
    )

    # Standardize confidence column
    if "confidence" in chunk.columns:
        chunk["confidence"] = chunk["confidence"].astype(str).str.lower()

    clean_rows += len(chunk)

    # Save cleaned chunk
    chunk.to_csv(
        OUTPUT_FILE,
        mode="w" if first_chunk else "a",
        header=first_chunk,
        index=False
    )

    first_chunk = False

    print(f"Processed {total_rows:,} rows | Saved {clean_rows:,} rows")

print("\n" + "=" * 60)
print("Cleaning Completed")
print("=" * 60)
print(f"Original Rows : {total_rows:,}")
print(f"Clean Rows    : {clean_rows:,}")
print(f"Saved File    : {OUTPUT_FILE}")