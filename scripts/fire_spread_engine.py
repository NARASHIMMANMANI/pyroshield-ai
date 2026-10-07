import pandas as pd
import numpy as np

print("=" * 60)
print("FIRE SPREAD ENGINE")
print("=" * 60)

# =====================================================
# Load Spread Dataset
# =====================================================

df = pd.read_csv("datasets/spread/spread_dataset.csv")

print("Dataset Loaded Successfully")
print("Shape :", df.shape)

# =====================================================
# Calculate Wind Speed
# =====================================================

if "wind_speed" in df.columns:
    wind_speed = df["wind_speed"]
else:
    wind_speed = np.sqrt(df["u_wind"]**2 + df["v_wind"]**2)

# =====================================================
# Load AI Wildfire Prediction
# =====================================================

prediction = pd.read_csv(
    "results/wildfire_prediction.csv"
)

# Safety Check

if len(prediction) != len(df):
    raise ValueError(
        "Prediction file and spread dataset have different number of rows."
    )

probability = prediction["fire_probability"]

# =====================================================
# Spread Speed
# =====================================================

spread_speed = (

    wind_speed * 0.30 +

    df["temperature"] * 0.05 +

    df["slope"] * 0.02 +

    probability * 3

)
spread_speed = spread_speed.clip(lower=0.1)

# =====================================================
# Spread Distance
# =====================================================

spread_distance = spread_speed * 2

# =====================================================
# Spread Direction
# =====================================================

angle = np.degrees(np.arctan2(df["v_wind"], df["u_wind"]))
angle = (angle + 360) % 360

direction = []

for a in angle:

    if a < 22.5 or a >= 337.5:
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
# Risk Level
# =====================================================

risk = []

for p, d in zip(probability, spread_distance):

    score = p * d * 1.2

    if score < 1:
        risk.append("LOW")

    elif score < 3:

        risk.append("MEDIUM")

    else:

        risk.append("HIGH")

# =====================================================
# Final Results
# =====================================================

results = pd.DataFrame({

    "latitude": df["latitude"],

    "longitude": df["longitude"],

    "fire_probability": probability,

    "spread_speed_kmh": spread_speed.round(2),

    "spread_distance_km": spread_distance.round(2),

    "spread_direction": direction,

    "risk_level": risk

})

# =====================================================
# Save
# =====================================================

results.to_csv(
    "results/fire_spread_forecast.csv",
    index=False
)

print("\nFire Spread Forecast Saved Successfully!")

print("\nLocation : results/fire_spread_forecast.csv")

print("\nPreview")

print(results.head())
print()

print("="*60)
print("SUMMARY")
print("="*60)

print("Average Fire Probability :",
      round(results["fire_probability"].mean(),3))

print("Average Spread Speed (km/h):",
      round(results["spread_speed_kmh"].mean(),2))

print("Average Spread Distance (km):",
      round(results["spread_distance_km"].mean(),2))

print()

print(results["risk_level"].value_counts())