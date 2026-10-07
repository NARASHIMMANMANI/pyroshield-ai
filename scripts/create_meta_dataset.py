import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import joblib
import torch
import pandas as pd
import numpy as np

from architectures.ft_transformer import FTTransformer
from architectures.kan_model import KANClassifier

# ==========================================================
# Load Dataset
# ==========================================================

print("=" * 60)
print("Creating Meta Dataset")
print("=" * 60)

df = pd.read_csv("datasets/final/ml_dataset.csv")

X = df.drop(columns=["fire"]).values.astype(np.float32)
y = df["fire"].values

# ==========================================================
# Random Forest
# ==========================================================

rf = joblib.load("models/random_forest.pkl")

rf_prob = rf.predict_proba(X)[:,1]

# ==========================================================
# XGBoost
# ==========================================================

xgb = joblib.load("models/xgboost.pkl")

xgb_prob = xgb.predict_proba(X)[:,1]

# ==========================================================
# CatBoost
# ==========================================================

cat = joblib.load("models/catboost.pkl")

cat_prob = cat.predict_proba(X)[:,1]

# ==========================================================
# FT Transformer
# ==========================================================

ft_scaler = joblib.load("models/ft_scaler.pkl")

X_ft = ft_scaler.transform(X)

ft_model = FTTransformer(num_features=X.shape[1])

ft_model.load_state_dict(
    torch.load("models/ft_transformer.pth",
               map_location="cpu")
)

ft_model.eval()

with torch.no_grad():

    ft_prob = torch.sigmoid(
        ft_model(torch.tensor(X_ft,dtype=torch.float32))
    ).numpy().flatten()

# ==========================================================
# KAN
# ==========================================================

kan_scaler = joblib.load("models/kan_scaler.pkl")

X_kan = kan_scaler.transform(X)

kan_model = KANClassifier(input_dim=X.shape[1])

kan_model.load_state_dict(
    torch.load("models/kan_model.pth",
               map_location="cpu")
)

kan_model.eval()

with torch.no_grad():

    kan_prob = torch.sigmoid(
        kan_model(torch.tensor(X_kan,dtype=torch.float32))
    ).numpy().flatten()

# ==========================================================
# Meta Dataset
# ==========================================================

meta = pd.DataFrame({

    "RandomForest":rf_prob,

    "XGBoost":xgb_prob,

    "CatBoost":cat_prob,

    "FTTransformer":ft_prob,

    "KAN":kan_prob,

    "fire":y

})

os.makedirs("datasets/meta",exist_ok=True)

meta.to_csv(
    "datasets/meta/meta_train.csv",
    index=False
)

print()

print(meta.head())

print()

print("Meta Dataset Saved Successfully!")

print(meta.shape)