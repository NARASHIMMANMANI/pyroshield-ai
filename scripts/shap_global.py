import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 60)
print("GLOBAL SHAP ANALYSIS")
print("=" * 60)

# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv("datasets/final/ml_dataset.csv")

X = df.drop(columns=["fire"])
y = df["fire"]

# ==========================================================
# Load Random Forest Model
# ==========================================================

model = joblib.load("models/random_forest.pkl")

# ==========================================================
# SHAP Explainer
# ==========================================================

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X)

# ==========================================================
# Create Output Folder
# ==========================================================

os.makedirs("results/shap", exist_ok=True)

# ==========================================================
# Summary Plot
# ==========================================================

plt.figure(figsize=(10, 7))

shap.summary_plot(
    shap_values[:, :, 1],
    X,
    show=False
)

plt.tight_layout()

plt.savefig(
    "results/shap/summary_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# Bar Plot
# ==========================================================

plt.figure(figsize=(10, 7))

shap.summary_plot(
    shap_values[:, :, 1],
    X,
    plot_type="bar",
    show=False
)

plt.tight_layout()

plt.savefig(
    "results/shap/bar_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()

print("Summary Plot Saved")

print("Bar Plot Saved")

print()

print("Location : results/shap/")