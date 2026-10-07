import pandas as pd
from pathlib import Path

# ==========================================================
# FILE PATHS
# ==========================================================

INPUT_FILE = "datasets/final/training_dataset.csv"
OUTPUT_FILE = "datasets/final/ml_dataset.csv"

Path("datasets/final").mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Wildfire Dataset Preprocessing")
print("=" * 60)

# ==========================================================
# LOAD DATASET
# ==========================================================

df = pd.read_csv(INPUT_FILE)

print(f"\nOriginal Shape : {df.shape}")

# ==========================================================
# REMOVE DUPLICATES
# ==========================================================

duplicates = df.duplicated().sum()
print(f"Duplicate Rows : {duplicates}")

df = df.drop_duplicates()

# ==========================================================
# DATE FEATURES
# ==========================================================

df["acq_date"] = pd.to_datetime(df["acq_date"])

df["year"] = df["acq_date"].dt.year
df["month"] = df["acq_date"].dt.month
df["day"] = df["acq_date"].dt.day

# ==========================================================
# DROP FIRE-ONLY COLUMNS
# These are not available before a fire occurs
# ==========================================================

drop_columns = [
    "brightness",
    "scan",
    "track",
    "acq_time",
    "satellite",
    "instrument",
    "confidence",
    "version",
    "bright_t31",
    "frp",
    "daynight",
    "type",
    "source",
    "landcover_name",
    "acq_date"
]

df = df.drop(columns=drop_columns, errors="ignore")

# ==========================================================
# HANDLE MISSING VALUES
# ==========================================================

print("\nMissing Values Before Cleaning")
print(df.isnull().sum())

# Fill numerical columns with median
numeric_columns = df.select_dtypes(include=["number"]).columns

for col in numeric_columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

# Remove any remaining missing values
df = df.dropna()

# ==========================================================
# FINAL FEATURE ORDER
# ==========================================================

features = [
    "latitude",
    "longitude",

    "temperature",
    "min_temperature",
    "max_temperature",
    "dewpoint",
    "precipitation",
    "surface_pressure",
    "sea_level_pressure",
    "u_wind",
    "v_wind",
    "wind_speed",

    "NDVI",
    "EVI",
    "NBR",
    "NDWI",

    "elevation",
    "slope",
    "aspect",

    "landcover_code",
    "population_density",
    "night_light",

    "year",
    "month",
    "day",

    "fire"
]

df = df[features]

# ==========================================================
# SAVE
# ==========================================================

df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 60)
print("Preprocessing Completed")
print("=" * 60)

print(f"Final Shape : {df.shape}")

print(f"\nSaved File : {OUTPUT_FILE}")

print("\nFinal Columns:")

for col in df.columns:
    print(col)