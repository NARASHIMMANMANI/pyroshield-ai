import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

print("=" * 60)
print("MANUAL CONFORMAL CALIBRATION")
print("=" * 60)

# ============================================================
# Load Meta Dataset
# ============================================================

df = pd.read_csv("datasets/meta/meta_train.csv")

X = df.drop(columns=["fire"])
y = df["fire"]

# ============================================================
# Split Calibration Data
# ============================================================

_, X_calib, _, y_calib = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# ============================================================
# Load Stacking Model
# ============================================================

model = joblib.load("models/stacking_model.pkl")

# ============================================================
# Predict Probabilities
# ============================================================

probs = model.predict_proba(X_calib)

# ============================================================
# Nonconformity Score
# score = 1 - probability(true class)
# ============================================================

scores = []

for i in range(len(y_calib)):
    true_class = int(y_calib.iloc[i])
    score = 1 - probs[i][true_class]
    scores.append(score)

scores = np.array(scores)

# ============================================================
# 95% Threshold
# ============================================================

threshold = np.quantile(scores, 0.95)

print()

print("Calibration Samples :", len(scores))
print("Threshold :", round(threshold, 4))

# ============================================================
# Save
# ============================================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    threshold,
    "models/conformal_threshold.pkl"
)

print()

print("Threshold Saved Successfully")