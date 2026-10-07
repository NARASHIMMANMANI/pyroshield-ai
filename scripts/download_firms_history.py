import os
import sys
import requests
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

# Project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import *

# Create output folder
os.makedirs(RAW_FIRMS_FOLDER, exist_ok=True)

current_date = datetime.strptime(START_DATE, "%Y-%m-%d")
end_date = datetime.strptime(END_DATE, "%Y-%m-%d")

all_data = []
while current_date <= end_date:

    batch_end = min(current_date + timedelta(days=4), end_date)

    start = current_date.strftime("%Y-%m-%d")
    stop = batch_end.strftime("%Y-%m-%d")

    print(f"\nDownloading {start} -> {stop}")

    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"{FIRMS_MAP_KEY}/"
        f"{DATASET}/"
        f"{AREA}/"
        f"{start}/{stop}"
    )

    response = requests.get(url)

    if response.status_code == 200:

        temp_file = os.path.join(
            RAW_FIRMS_FOLDER,
            f"{start}.csv"
        )

        with open(temp_file, "wb") as f:
            f.write(response.content)

        df = pd.read_csv(temp_file)

        print("Rows:", len(df))

        all_data.append(df)

    else:

        print("Failed:", response.status_code)
        print(response.text)

    current_date += timedelta(days=5)
if len(all_data):

    final_df = pd.concat(all_data)

    final_df.drop_duplicates(inplace=True)

    output = os.path.join(
        RAW_FIRMS_FOLDER,
        "fire_events.csv"
    )

    final_df.to_csv(output, index=False)

    print("\nFinished!")

    print("Total Records:", len(final_df))

    print("Saved to:")
    print(output)

else:

    print("No data downloaded.")