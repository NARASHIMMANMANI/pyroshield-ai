import os
import sys
import joblib
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from architectures.ft_transformer import FTTransformer
from architectures.kan_model import KANClassifier
print("="*60)
print("PYROSHIELD-AI WILDFIRE PREDICTION")
print("="*60)

df = pd.read_csv("datasets/final/ml_dataset.csv")

X = df.drop(columns=["fire"]).copy()
rf = joblib.load("models/random_forest.pkl")
xgb = joblib.load("models/xgboost.pkl")
cat = joblib.load("models/catboost.pkl")
ft_scaler = joblib.load("models/ft_scaler.pkl")
kan_scaler = joblib.load("models/kan_scaler.pkl")
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device :", device)
ft_model = FTTransformer(
    num_features=X.shape[1]
)

ft_model.load_state_dict(
    torch.load(
        "models/ft_transformer.pth",
        map_location=device
    )
)

ft_model.to(device)
ft_model.eval()
kan_model = KANClassifier(
    input_dim=X.shape[1]
)

kan_model.load_state_dict(
    torch.load(
        "models/kan_model.pth",
        map_location=device
    )
)

kan_model.to(device)
kan_model.eval()
rf_prob = rf.predict_proba(X)[:,1]
xgb_prob = xgb.predict_proba(X)[:,1]
cat_prob = cat.predict_proba(X)[:,1]
X_ft = ft_scaler.transform(X)

X_ft = torch.tensor(
    X_ft,
    dtype=torch.float32
).to(device)

with torch.no_grad():

    ft_prob = torch.sigmoid(
        ft_model(X_ft).squeeze()
    ).cpu().numpy()
    X_kan = kan_scaler.transform(X)

X_kan = torch.tensor(
    X_kan,
    dtype=torch.float32
).to(device)

with torch.no_grad():

    kan_prob = torch.sigmoid(
        kan_model(X_kan).squeeze()
    ).cpu().numpy()
    meta_features = pd.DataFrame({

    "RandomForest": rf_prob,

    "XGBoost": xgb_prob,

    "CatBoost": cat_prob,

    "FTTransformer": ft_prob,

    "KAN": kan_prob

})

print(meta_features.head())
# =====================================================
# Load Stacking Model
# =====================================================

stacking_model = joblib.load("models/stacking_model.pkl")

# =====================================================
# Final Wildfire Probability
# =====================================================

final_probability = stacking_model.predict_proba(meta_features)[:, 1]

final_prediction = (final_probability >= 0.5).astype(int)

# =====================================================
# Create Final Output
# =====================================================

output = pd.DataFrame({
    "latitude": df["latitude"],
    "longitude": df["longitude"],
    "fire_probability": np.round(final_probability, 6),
    "fire_prediction": final_prediction
})

# =====================================================
# Save Results
# =====================================================

os.makedirs("results", exist_ok=True)

output.to_csv(
    "results/wildfire_prediction.csv",
    index=False
)

print("\nFinal Wildfire Predictions Saved Successfully!")
print("Location : results/wildfire_prediction.csv")

print("\nPreview")
print(output.head())