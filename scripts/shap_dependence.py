import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 60)
print("SHAP DEPENDENCE PLOTS")
print("=" * 60)

# =====================================================
# Load Dataset
# =====================================================

df = pd.read_csv("datasets/final/ml_dataset.csv")

X = df.drop(columns=["fire"])

# =====================================================
# Load Model
# =====================================================

model = joblib.load("models/random_forest.pkl")

# =====================================================
# SHAP Values
# =====================================================

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X)

# Binary classification
values = shap_values[:, :, 1]

os.makedirs("results/shap", exist_ok=True)

# =====================================================
# Temperature
# =====================================================

plt.figure(figsize=(8,6))

shap.dependence_plot(
    "temperature",
    values,
    X,
    show=False
)

plt.tight_layout()

plt.savefig(
    "results/shap/dependence_temperature.png",
    dpi=300
)

plt.close()

# =====================================================
# NDVI
# =====================================================

plt.figure(figsize=(8,6))

shap.dependence_plot(
    "NDVI",
    values,
    X,
    show=False
)

plt.tight_layout()

plt.savefig(
    "results/shap/dependence_ndvi.png",
    dpi=300
)

plt.close()

# =====================================================
# Wind Speed
# =====================================================

plt.figure(figsize=(8,6))

shap.dependence_plot(
    "wind_speed",
    values,
    X,
    show=False
)

plt.tight_layout()

plt.savefig(
    "results/shap/dependence_windspeed.png",
    dpi=300
)

plt.close()

print()

print("Temperature Plot Saved")

print("NDVI Plot Saved")

print("Wind Speed Plot Saved")

print()

print("Location : results/shap/")