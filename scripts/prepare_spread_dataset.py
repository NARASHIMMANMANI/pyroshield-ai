import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 60)
print("PREPARING FIRE SPREAD DATASET")
print("=" * 60)

# =====================================================
# Load Dataset
# =====================================================

df = pd.read_csv("datasets/final/ml_dataset.csv")

print("Original Shape :", df.shape)

# =====================================================
# Wind Speed
# =====================================================

if "wind_speed" in df.columns:
    wind = df["wind_speed"]
else:
    wind = np.sqrt(df["u_wind"]**2 + df["v_wind"]**2)

# =====================================================
# Spread Speed (km/hr)
# =====================================================

spread_speed = (
    0.30 * wind +
    0.05 * (df["temperature"] - 20) +
    0.02 * df["slope"]
)

spread_speed = spread_speed.clip(lower=0.1)

# =====================================================
# Spread Distance (km)
# =====================================================

spread_distance = spread_speed * 2.0

# =====================================================
# Spread Direction
# =====================================================

angle = np.degrees(np.arctan2(df["v_wind"], df["u_wind"]))
angle = (angle + 360) % 360

direction = []

for a in angle:

    if 337.5 <= a or a < 22.5:
        direction.append("East")

    elif a < 67.5:
        direction.append("North-East")

    elif a < 112.5:
        direction.append("North")

    elif a < 157.5:
        direction.append("North-West")

    elif a < 202.5:
        direction.append("West")

    elif a < 247.5:
        direction.append("South-West")

    elif a < 292.5:
        direction.append("South")

    else:
        direction.append("South-East")

# =====================================================
# Risk Zone
# =====================================================

risk = []

for d in spread_distance:

    if d < 2:
        risk.append("LOW")

    elif d < 5:
        risk.append("MEDIUM")

    else:
        risk.append("HIGH")

# =====================================================
# Append Columns
# =====================================================

df["spread_speed"] = spread_speed.round(2)

df["spread_distance"] = spread_distance.round(2)

df["spread_direction"] = direction

df["risk_zone"] = risk

# =====================================================
# Save
# =====================================================

Path("datasets/spread").mkdir(
    parents=True,
    exist_ok=True
)

output_path = "datasets/spread/spread_dataset.csv"

df.to_csv(
    output_path,
    index=False
)

print("\nSpread Dataset Created Successfully!")

print("Location :", output_path)

print("\nDataset Shape :", df.shape)

print("\nPreview")

print(df[
    [
        "spread_speed",
        "spread_distance",
        "spread_direction",
        "risk_zone"
    ]
].head())