import pandas as pd
from pathlib import Path

from modules.weather import get_weather
from modules.vegetation import get_vegetation
from modules.terrain import get_terrain
from modules.landcover import get_landcover
from modules.population import get_population
from modules.nightlights import get_nightlights

INPUT_FILE = "datasets/processed/sample_100.csv"
OUTPUT_FILE = "datasets/final/wildfire_features.csv"

Path("datasets/final").mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

results = []

print("=" * 60)
print("Feature Extraction Started")
print("=" * 60)

for i, row in df.iterrows():

    lat = row["latitude"]
    lon = row["longitude"]
    date = row["acq_date"]

    print(f"[{i+1}/{len(df)}] Processing ({lat}, {lon})")

    try:

        weather = get_weather(lat, lon, date)
        vegetation = get_vegetation(lat, lon, date)
        terrain = get_terrain(lat, lon)
        landcover = get_landcover(lat, lon)
        population = get_population(lat, lon)
        night = get_nightlights(lat, lon, date)

        new_row = row.to_dict()

        new_row.update(weather)
        new_row.update(vegetation)
        new_row.update(terrain)
        new_row.update(landcover)
        new_row.update(population)
        new_row.update(night)

        results.append(new_row)

    except Exception as e:

        print(f"Skipped row {i+1}: {e}")

output = pd.DataFrame(results)

output.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 60)
print("Feature Extraction Completed")
print("=" * 60)
print(f"Rows Processed : {len(output)}")
print(f"Saved File     : {OUTPUT_FILE}")