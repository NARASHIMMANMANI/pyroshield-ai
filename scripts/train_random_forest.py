import pandas as pd
import joblib
import os
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

print("=" * 60)
print("Random Forest Training")
print("=" * 60)

# =====================================================
# Load Dataset
# =====================================================

df = pd.read_csv("datasets/final/ml_dataset.csv")

X = df.drop(columns=["fire"])
y = df["fire"]

print(f"Dataset Shape : {df.shape}")
print(f"Features      : {X.shape[1]}")

# =====================================================
# Train/Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")

# =====================================================
# Model
# =====================================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# =====================================================
# Prediction
# =====================================================

prediction = model.predict(X_test)
probability = model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, prediction)
precision = precision_score(y_test, prediction)
recall = recall_score(y_test, prediction)
f1 = f1_score(y_test, prediction)
roc = roc_auc_score(y_test, probability)
# =====================================================
# Evaluation
# =====================================================

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"ROC AUC  : {roc:.4f}")

print("\nClassification Report")
print(classification_report(y_test, prediction))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, prediction))

# =====================================================
# Feature Importance
# =====================================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Features")
print(importance.head(10))

# =====================================================
# Create Folders
# =====================================================

Path("results").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

# =====================================================
# Save Feature Importance
# =====================================================

importance.to_csv(
    "results/feature_importance_rf.csv",
    index=False
)

# =====================================================
# Save Model
# =====================================================

joblib.dump(
    model,
    "models/random_forest.pkl"
)

print("\nFeature Importance Saved : results/feature_importance_rf.csv")
print("Model Saved             : models/random_forest.pkl")

print("\nRandom Forest Training Completed Successfully!")
# ==========================================================
# Save Metrics
# ==========================================================

metrics = pd.DataFrame({
    "Model": ["Random Forest"],
    "Accuracy": [round(accuracy, 4)],
    "Precision": [round(precision, 4)],
    "Recall": [round(recall, 4)],
    "F1 Score": [round(f1, 4)],
    "ROC AUC": [round(roc, 4)]
})

metrics.to_csv(
    "results/random_forest_metrics.csv",
    index=False
)

print("Random Forest Metrics Saved Successfully!")