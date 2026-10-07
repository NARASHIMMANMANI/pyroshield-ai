import pandas as pd
import joblib
from pathlib import Path

from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

print("="*60)
print("LightGBM Training")
print("="*60)

df = pd.read_csv("datasets/final/ml_dataset.csv")

X = df.drop(columns=["fire"])
y = df["fire"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

prediction = model.predict(X_test)
probability = model.predict_proba(X_test)[:, 1]

print("\nAccuracy :", accuracy_score(y_test, prediction))
print("\nROC AUC :", roc_auc_score(y_test, probability))

print("\nClassification Report")
print(classification_report(y_test, prediction))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, prediction))

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

Path("results").mkdir(exist_ok=True)

importance.to_csv(
    "results/feature_importance_lightgbm.csv",
    index=False
)

Path("models").mkdir(exist_ok=True)

joblib.dump(
    model,
    "models/lightgbm.pkl"
)

print("\nTop Features")
print(importance.head(10))

print("\nModel Saved.")