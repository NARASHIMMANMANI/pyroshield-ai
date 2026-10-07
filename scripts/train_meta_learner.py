import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

print("=" * 60)
print("META LEARNER TRAINING")
print("=" * 60)

# =====================================================
# Load Meta Dataset
# =====================================================

df = pd.read_csv("datasets/meta/meta_train.csv")

X = df.drop(columns=["fire"])
y = df["fire"]

# =====================================================
# Train Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =====================================================
# Logistic Regression
# =====================================================

model = LogisticRegression(
    random_state=42,
    max_iter=1000
)

model.fit(X_train, y_train)

# =====================================================
# Prediction
# =====================================================

prediction = model.predict(X_test)

probability = model.predict_proba(X_test)[:, 1]

# =====================================================
# Metrics
# =====================================================

accuracy = accuracy_score(y_test, prediction)
precision = precision_score(y_test, prediction)
recall = recall_score(y_test, prediction)
f1 = f1_score(y_test, prediction)
roc = roc_auc_score(y_test, probability)

print()

print("=" * 60)
print("STACKING RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {roc:.4f}")

print()

print("Classification Report")

print(classification_report(y_test, prediction))

print()

print("Confusion Matrix")

print(confusion_matrix(y_test, prediction))

# =====================================================
# Save Model
# =====================================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/stacking_model.pkl"
)

print()

print("Meta Learner Saved Successfully!")