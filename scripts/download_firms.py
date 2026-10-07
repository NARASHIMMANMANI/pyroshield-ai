import os
import sys
import requests
import pandas as pd

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import *

# Create output folder
os.makedirs(RAW_FIRMS_FOLDER, exist_ok=True)


url = f"https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv/{FIRMS_MAP_KEY}/ALL"

print("=" * 60)
print("Downloading NASA FIRMS Fire Events...")
print("=" * 60)

response = requests.get(url)

print("Status Code:", response.status_code)

if response.status_code == 200:

    output_file = os.path.join(RAW_FIRMS_FOLDER, "fire_events.csv")

    with open(output_file, "wb") as f:
        f.write(response.content)

    df = pd.read_csv(output_file)

    print("\nDownload Successful!")
    print("Rows :", len(df))
    print("Columns :", len(df.columns))
    print(df.head())

else:
    print("Download Failed")
    print(response.text)