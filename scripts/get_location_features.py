import os
import numpy as np
import pandas as pd


DATASET_PATH = "datasets/final/ml_dataset.csv"


FEATURE_COLUMNS = [
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
]


def get_location_features(latitude, longitude):

    # -------------------------------------------------
    # Load dataset
    # -------------------------------------------------

    df = pd.read_csv(DATASET_PATH)

    # -------------------------------------------------
    # Calculate geographical distance
    # -------------------------------------------------

    distance = (
        (df["latitude"] - latitude) ** 2
        +
        (df["longitude"] - longitude) ** 2
    )

    nearest_index = distance.idxmin()

    row = df.loc[nearest_index]

    # -------------------------------------------------
    # Create feature DataFrame
    # -------------------------------------------------

    sample = pd.DataFrame(
        [[row[column] for column in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS
    )

    # -------------------------------------------------
    # IMPORTANT:
    # Keep the user's coordinates
    # -------------------------------------------------

    sample.loc[0, "latitude"] = latitude
    sample.loc[0, "longitude"] = longitude

    return sample


# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":

    print("=" * 60)
    print("LOCATION FEATURE RETRIEVAL TEST")
    print("=" * 60)

    latitude = float(
        input("Enter Latitude  : ")
    )

    longitude = float(
        input("Enter Longitude : ")
    )

    sample = get_location_features(
        latitude,
        longitude
    )

    print("\n25 FEATURES")
    print("=" * 60)

    print(sample.T)

    print("\nNumber of Features :", sample.shape[1])

    print("\nFeature Retrieval Successful!")