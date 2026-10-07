import pandas as pd
import numpy as np
from pathlib import Path

# =====================================================
# Configuration
# =====================================================

INPUT_FILE = "datasets/processed/sample_100.csv"      # Change later to wildfire_cleaned.csv
OUTPUT_FILE = "datasets/processed/nonfire_samples.csv"

NUM_SAMPLES = 100          # Generate same number as fire samples
RANDOM_SEED = 42

# =====================================================
# Load Fire Dataset
# =====================================================

print("=" * 60)
print("Generating Non-Fire Samples")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

np.random.seed(RANDOM_SEED)

# =====================================================
# Bounding Box
# =====================================================

min_lat = df["latitude"].min()
max_lat = df["latitude"].max()

min_lon = df["longitude"].min()
max_lon = df["longitude"].max()

print(f"Latitude : {min_lat:.4f} -> {max_lat:.4f}")
print(f"Longitude: {min_lon:.4f} -> {max_lon:.4f}")

# =====================================================
# Existing Fire Locations
# =====================================================

fire_locations = set(
    zip(
        df["latitude"].round(4),
        df["longitude"].round(4)
    )
)

nonfire = []

# =====================================================
# Generate Random Locations
# =====================================================

while len(nonfire) < NUM_SAMPLES:

    lat = np.random.uniform(min_lat, max_lat)
    lon = np.random.uniform(min_lon, max_lon)

    key = (round(lat, 4), round(lon, 4))

    if key in fire_locations:
        continue

    nonfire.append({
        "latitude": lat,
        "longitude": lon,
        "acq_date": np.random.choice(df["acq_date"]),
        "fire": 0
    })

# =====================================================
# Save
# =====================================================

nonfire_df = pd.DataFrame(nonfire)

Path("datasets/processed").mkdir(parents=True, exist_ok=True)

nonfire_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nGenerated Successfully!")
print(nonfire_df.head())

print(f"\nSaved to: {OUTPUT_FILE}")
print(f"Total Non-Fire Samples: {len(nonfire_df)}")