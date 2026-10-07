import pandas as pd
from pathlib import Path
import joblib

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

print("="*60)
print("XGBoost Training")
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

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

prediction = model.predict(X_test)
probability = model.predict_proba(X_test)[:,1]
accuracy = accuracy_score(y_test, prediction)
precision = precision_score(y_test, prediction)
recall = recall_score(y_test, prediction)
f1 = f1_score(y_test, prediction)
roc = roc_auc_score(y_test, probability)

print("\nAccuracy :", round(accuracy,4))
print("\nROC AUC :", round(roc,4))

print("\nClassification Report")
print(classification_report(y_test,prediction))

print("\nConfusion Matrix")
print(confusion_matrix(y_test,prediction))

importance = pd.DataFrame({
    "Feature":X.columns,
    "Importance":model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

Path("results").mkdir(exist_ok=True)

importance.to_csv(
    "results/feature_importance_xgboost.csv",
    index=False
)

Path("models").mkdir(exist_ok=True)

joblib.dump(
    model,
    "models/xgboost.pkl"
)

print("\nTop Features")

print(importance.head(10))

print("\nModel Saved.")
# ==========================================================
# Save Metrics
# ==========================================================

metrics = pd.DataFrame({

    "Model": ["XGBoost"],

    "Accuracy": [round(accuracy,4)],

    "Precision": [round(precision,4)],

    "Recall": [round(recall,4)],

    "F1 Score": [round(f1,4)],

    "ROC AUC": [round(roc,4)]

})

metrics.to_csv(

    "results/xgboost_metrics.csv",

    index=False

)

print("XGBoost Metrics Saved Successfully!")