import os
import joblib
import pandas as pd

print("=" * 60)
print("MANUAL CONFORMAL PREDICTION")
print("=" * 60)

# ============================================================
# Load Data
# ============================================================

df = pd.read_csv("datasets/meta/meta_train.csv")

X = df.drop(columns=["fire"])

# ============================================================
# Load Model
# ============================================================

model = joblib.load("models/stacking_model.pkl")

threshold = joblib.load(
    "models/conformal_threshold.pkl"
)

# ============================================================
# Predict
# ============================================================

probs = model.predict_proba(X)

predictions = model.predict(X)

results = []

for pred, prob in zip(predictions, probs):

    confidence = prob[pred]

    reliable = confidence >= (1 - threshold)

    results.append([
        pred,
        round(confidence, 4),
        reliable
    ])

results = pd.DataFrame(
    results,
    columns=[
        "Prediction",
        "Confidence",
        "Reliable"
    ]
)

print()

print(results.head())

# ============================================================
# Save
# ============================================================

os.makedirs(
    "results/uncertainty",
    exist_ok=True
)

results.to_csv(
    "results/uncertainty/conformal_predictions.csv",
    index=False
)

print()

print("Results Saved Successfully")