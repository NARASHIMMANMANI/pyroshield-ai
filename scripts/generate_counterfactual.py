import os
import joblib
import pandas as pd
import dice_ml

print("=" * 60)
print("GENERATING COUNTERFACTUAL EXPLANATIONS")
print("=" * 60)

# ============================================================
# Load Dataset
# ============================================================

df = pd.read_csv("datasets/final/ml_dataset.csv")

# ============================================================
# Load DiCE Objects
# ============================================================

data = joblib.load("models/dice_data.pkl")

model = joblib.load("models/dice_model.pkl")

# ============================================================
# Create DiCE Explainer
# ============================================================

explainer = dice_ml.Dice(
    data,
    model,
    method="random"
)

# ============================================================
# Select Sample
# ============================================================

query = df[df["fire"] == 1].drop(columns=["fire"]).iloc[[0]]

print()

print("Original Sample")

print(query)

print()

# ============================================================
# Generate Counterfactuals
# ============================================================

cf = explainer.generate_counterfactuals(
    query,
    total_CFs=3,
    desired_class="opposite"
)

# ============================================================
# Save
# ============================================================

os.makedirs(
    "results/counterfactual",
    exist_ok=True
)

cf.visualize_as_dataframe()

cf.cf_examples_list[0].final_cfs_df.to_csv(
    "results/counterfactual/counterfactuals.csv",
    index=False
)

print()

print("Counterfactuals Saved Successfully!")

print()

print("Location : results/counterfactual/")