import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 60)
print("LOCAL SHAP EXPLANATION")
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
# SHAP Explainer
# =====================================================

explainer = shap.TreeExplainer(model)

shap_values = explainer(X)

# =====================================================
# Create Output Folder
# =====================================================

os.makedirs("results/shap", exist_ok=True)

# =====================================================
# Waterfall Plot (First Sample)
# =====================================================

plt.figure(figsize=(10, 8))

shap.plots.waterfall(
    shap_values[0, :, 1],
    show=False
)

plt.savefig(
    "results/shap/waterfall_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()

print("Waterfall Plot Saved Successfully!")

print("Location : results/shap/")