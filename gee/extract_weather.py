import ee
import pandas as pd
import os

# ---------------------------------
# Initialize Earth Engine
# ---------------------------------
ee.Initialize(project="erudite-river-502705-n5")

# ---------------------------------
# File paths
# ---------------------------------
INPUT_CSV = "datasets/raw/firms/sample_fire_events.csv"
OUTPUT_CSV = "datasets/processed/weather_features.csv"

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# ---------------------------------
# Read fire events
# ---------------------------------
df = pd.read_csv(INPUT_CSV)

results = []

# ---------------------------------
# Process each fire point
# ---------------------------------
for _, row in df.iterrows():

    lat = row["latitude"]
    lon = row["longitude"]
    date = row["date"]

    point = ee.Geometry.Point([lon, lat])

    start = ee.Date(date)
    end = start.advance(1, "day")

    era5 = (
        ee.ImageCollection("ECMWF/ERA5/DAILY")
        .filterDate(start, end)
        .first()
    )

    data = era5.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=point,
        scale=10000,
        maxPixels=1e9
    ).getInfo()

    results.append({
        "latitude": lat,
        "longitude": lon,
        "date": date,
        "fire": row["fire"],

        "mean_2m_air_temperature": data.get("mean_2m_air_temperature"),
        "minimum_2m_air_temperature": data.get("minimum_2m_air_temperature"),
        "maximum_2m_air_temperature": data.get("maximum_2m_air_temperature"),

        "total_precipitation": data.get("total_precipitation"),

        "u_component_of_wind_10m": data.get("u_component_of_wind_10m"),

        "v_component_of_wind_10m": data.get("v_component_of_wind_10m"),
    })

# ---------------------------------
# Save results
# ---------------------------------
weather_df = pd.DataFrame(results)

weather_df.to_csv(OUTPUT_CSV, index=False)

print("\nWeather feature extraction completed.")
print(weather_df.head())
print(f"\nSaved to: {OUTPUT_CSV}")