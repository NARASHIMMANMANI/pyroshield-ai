import os
import sys
import joblib
import pandas as pd
import numpy as np
import torch
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.append(PROJECT_ROOT)

from architectures.ft_transformer import FTTransformer
from architectures.kan_model import KANClassifier


# =====================================================
# CONFIGURATION
# =====================================================

DATASET_PATH = "datasets/final/ml_dataset.csv"

RF_PATH = "models/random_forest.pkl"
XGB_PATH = "models/xgboost.pkl"
CAT_PATH = "models/catboost.pkl"
STACK_PATH = "models/stacking_model.pkl"

FT_SCALER_PATH = "models/ft_scaler.pkl"
KAN_SCALER_PATH = "models/kan_scaler.pkl"

FT_MODEL_PATH = "models/ft_transformer.pth"
KAN_MODEL_PATH = "models/kan_model.pth"

OUTPUT_PATH = "results/single_location_prediction.csv"


# =====================================================
# HEADER
# =====================================================

print("=" * 60)
print("                 PYROSHIELD-AI")
print("=" * 60)


# =====================================================
# LOAD DATASET
# =====================================================

dataset = pd.read_csv(DATASET_PATH)


# =====================================================
# USER INPUT
# =====================================================

try:

    latitude = float(input("\nEnter Latitude  : "))
    longitude = float(input("Enter Longitude : "))

except ValueError:

    print("\nInvalid latitude or longitude.")
    sys.exit()


# =====================================================
# VALIDATE COORDINATES
# =====================================================

if latitude < -90 or latitude > 90:

    print("\nInvalid latitude. Latitude must be between -90 and 90.")
    sys.exit()


if longitude < -180 or longitude > 180:

    print("\nInvalid longitude. Longitude must be between -180 and 180.")
    sys.exit()


# =====================================================
# FIND NEAREST DATASET LOCATION
# =====================================================

dataset["distance"] = (
    (dataset["latitude"] - latitude) ** 2 +
    (dataset["longitude"] - longitude) ** 2
)

nearest_index = dataset["distance"].idxmin()

nearest = dataset.loc[nearest_index]

dataset_latitude = nearest["latitude"]
dataset_longitude = nearest["longitude"]


# =====================================================
# CALCULATE ACTUAL GEOGRAPHICAL DISTANCE
# HAVERSINE FORMULA
# =====================================================

earth_radius_km = 6371.0

lat1 = np.radians(latitude)
lat2 = np.radians(dataset_latitude)

delta_lat = np.radians(dataset_latitude - latitude)
delta_lon = np.radians(dataset_longitude - longitude)

a = (
    np.sin(delta_lat / 2) ** 2
    +
    np.cos(lat1)
    * np.cos(lat2)
    * np.sin(delta_lon / 2) ** 2
)

c = 2 * np.arctan2(
    np.sqrt(a),
    np.sqrt(1 - a)
)

distance_km = earth_radius_km * c


# =====================================================
# PREPARE SAMPLE
# =====================================================

sample = nearest.drop(
    labels=["fire", "distance"]
)

sample = sample.to_frame().T


# =====================================================
# DEVICE
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =====================================================
# LOAD CLASSICAL MODELS
# =====================================================

rf_model = joblib.load(RF_PATH)

xgb_model = joblib.load(XGB_PATH)

cat_model = joblib.load(CAT_PATH)

stack_model = joblib.load(STACK_PATH)


# =====================================================
# LOAD SCALERS
# =====================================================

ft_scaler = joblib.load(FT_SCALER_PATH)

kan_scaler = joblib.load(KAN_SCALER_PATH)


# =====================================================
# LOAD FT-TRANSFORMER
# =====================================================

ft_model = FTTransformer(
    num_features=25
)

ft_model.load_state_dict(
    torch.load(
        FT_MODEL_PATH,
        map_location=device
    )
)

ft_model.to(device)

ft_model.eval()


# =====================================================
# LOAD KAN
# =====================================================

kan_model = KANClassifier(
    input_dim=25
)

kan_model.load_state_dict(
    torch.load(
        KAN_MODEL_PATH,
        map_location=device
    )
)

kan_model.to(device)

kan_model.eval()


# =====================================================
# RANDOM FOREST
# =====================================================

rf_probability = rf_model.predict_proba(
    sample
)[0][1]


# =====================================================
# XGBOOST
# =====================================================

xgb_probability = xgb_model.predict_proba(
    sample
)[0][1]


# =====================================================
# CATBOOST
# =====================================================

cat_probability = cat_model.predict_proba(
    sample
)[0][1]


# =====================================================
# FT-TRANSFORMER
# =====================================================

sample_ft = ft_scaler.transform(sample)

sample_ft = torch.tensor(
    sample_ft,
    dtype=torch.float32
).to(device)


with torch.no_grad():

    ft_probability = torch.sigmoid(
        ft_model(sample_ft)
    ).item()


# =====================================================
# KAN
# =====================================================

sample_kan = kan_scaler.transform(sample)

sample_kan = torch.tensor(
    sample_kan,
    dtype=torch.float32
).to(device)


with torch.no_grad():

    kan_probability = torch.sigmoid(
        kan_model(sample_kan)
    ).item()


# =====================================================
# STACKING ENSEMBLE
# =====================================================

meta_features = pd.DataFrame({

    "RandomForest": [rf_probability],

    "XGBoost": [xgb_probability],

    "CatBoost": [cat_probability],

    "FTTransformer": [ft_probability],

    "KAN": [kan_probability]

})


final_probability = stack_model.predict_proba(
    meta_features
)[0][1]


final_prediction = stack_model.predict(
    meta_features
)[0]


# =====================================================
# FIRE SPREAD ESTIMATION
# =====================================================

wind_speed = sample["wind_speed"].iloc[0]

temperature = sample["temperature"].iloc[0]

slope = sample["slope"].iloc[0]


spread_speed = (

    wind_speed * 0.30

    + temperature * 0.05

    + slope * 0.02

    + final_probability * 3

)


spread_speed = max(
    spread_speed,
    0.1
)


spread_distance = spread_speed * 2


# =====================================================
# FIRE SPREAD DIRECTION
# =====================================================

u = sample["u_wind"].iloc[0]

v = sample["v_wind"].iloc[0]


angle = np.degrees(
    np.arctan2(v, u)
)

angle = (
    angle + 360
) % 360


if angle < 22.5 or angle >= 337.5:

    direction = "East"

elif angle < 67.5:

    direction = "North-East"

elif angle < 112.5:

    direction = "North"

elif angle < 157.5:

    direction = "North-West"

elif angle < 202.5:

    direction = "West"

elif angle < 247.5:

    direction = "South-West"

elif angle < 292.5:

    direction = "South"

else:

    direction = "South-East"


# =====================================================
# RISK LEVEL
# =====================================================

risk_score = (
    final_probability
    * spread_distance
    * 1.2
)


if risk_score < 1:

    risk_level = "LOW"

elif risk_score < 3:

    risk_level = "MEDIUM"

else:

    risk_level = "HIGH"


# =====================================================
# ALERT LEVEL
# =====================================================

if risk_level == "HIGH":

    alert = "RED"

    priority = "CRITICAL"

    monitoring = "Every 15 Minutes"


elif risk_level == "MEDIUM":

    alert = "ORANGE"

    priority = "HIGH"

    monitoring = "Every 30 Minutes"


else:

    alert = "GREEN"

    priority = "LOW"

    monitoring = "Every 2 Hours"


# =====================================================
# RECOMMENDATION
# =====================================================

if alert == "RED":

    recommendation = [

        "Deploy firefighters immediately",

        "Launch drone surveillance",

        "Notify disaster management authority",

        "Continuous monitoring every 15 minutes"

    ]


elif alert == "ORANGE":

    recommendation = [

        "Increase surveillance",

        "Prepare nearby fire stations",

        "Monitor weather conditions"

    ]


else:

    recommendation = [

        "Routine monitoring",

        "No emergency response required"

    ]


# =====================================================
# USER VISIBLE FINAL OUTPUT
# =====================================================

print()
print("=" * 60)
print("              PYROSHIELD-AI RESULT")
print("=" * 60)

print(
    f"Input Location       : "
    f"{latitude:.6f}, {longitude:.6f}"
)

print(
    f"Dataset Location     : "
    f"{dataset_latitude:.6f}, {dataset_longitude:.6f}"
)

print(
    f"Dataset Distance     : "
    f"{distance_km:.2f} km"
)

print()

print(
    f"Fire Probability     : "
    f"{final_probability * 100:.2f}%"
)

if final_prediction == 1:

    print("Prediction           : 🔥 FIRE")

else:

    print("Prediction           : ✅ NO FIRE")

print()

print(
    f"Spread Speed         : "
    f"{spread_speed:.2f} km/h"
)

print(
    f"Spread Distance      : "
    f"{spread_distance:.2f} km"
)

print(
    f"Spread Direction     : "
    f"{direction}"
)

print()

print(
    f"Risk Level           : "
    f"{risk_level}"
)

print(
    f"Alert Level          : "
    f"{alert}"
)

print(
    f"Priority             : "
    f"{priority}"
)

print(
    f"Monitoring           : "
    f"{monitoring}"
)

print()

print("Recommended Actions")

for i, action in enumerate(
    recommendation,
    start=1
):

    print(
        f"{i}. {action}"
    )

print("=" * 60)


# =====================================================
# SAVE RESULT
# =====================================================

os.makedirs(
    "results",
    exist_ok=True
)


report = pd.DataFrame({

    "input_latitude": [
        latitude
    ],

    "input_longitude": [
        longitude
    ],

    "dataset_latitude": [
        dataset_latitude
    ],

    "dataset_longitude": [
        dataset_longitude
    ],

    "dataset_distance_km": [
        round(distance_km, 3)
    ],

    "temperature": [
        sample["temperature"].iloc[0]
    ],

    "wind_speed": [
        sample["wind_speed"].iloc[0]
    ],

    "fire_probability": [
        round(
            final_probability,
            4
        )
    ],

    "prediction": [

        "FIRE"
        if final_prediction == 1
        else "NO FIRE"

    ],

    "spread_speed_kmh": [

        round(
            spread_speed,
            2
        )

    ],

    "spread_distance_km": [

        round(
            spread_distance,
            2
        )

    ],

    "spread_direction": [

        direction

    ],

    "risk_level": [

        risk_level

    ],

    "alert_level": [

        alert

    ],

    "priority": [

        priority

    ],

    "monitoring": [

        monitoring

    ]

})


report.to_csv(
    OUTPUT_PATH,
    index=False
)