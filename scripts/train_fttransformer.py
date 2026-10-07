import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from architectures.ft_transformer import FTTransformer


print("=" * 60)
print("FT-TRANSFORMER TRAINING")
print("=" * 60)


# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv("datasets/final/ml_dataset.csv")

X = df.drop(columns=["fire"]).values.astype(np.float32)
y = df["fire"].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# ==========================================================
# Feature Scaling
# ==========================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "models/ft_scaler.pkl")


# ==========================================================
# Dataset Class
# ==========================================================

class WildfireDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]


train_dataset = WildfireDataset(X_train, y_train)
test_dataset = WildfireDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)


# ==========================================================
# Device
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# ==========================================================
# Model
# ==========================================================

model = FTTransformer(
    num_features=X_train.shape[1]
)

model.to(device)


criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001
)


# ==========================================================
# Training
# ==========================================================

epochs = 50

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for features, labels in train_loader:

        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(features).squeeze()

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch {epoch+1:02d}/{epochs} "
        f"Loss={running_loss/len(train_loader):.4f}"
    )


# ==========================================================
# Evaluation
# ==========================================================

model.eval()

predictions = []
probabilities = []
targets = []

with torch.no_grad():

    for features, labels in test_loader:

        features = features.to(device)

        outputs = model(features).squeeze()

        probs = torch.sigmoid(outputs)

        preds = (probs >= 0.5).float()

        predictions.extend(preds.cpu().numpy())

        probabilities.extend(probs.cpu().numpy())

        targets.extend(labels.numpy())


accuracy = accuracy_score(targets, predictions)
precision = precision_score(targets, predictions)
recall = recall_score(targets, predictions)
f1 = f1_score(targets, predictions)
roc = roc_auc_score(targets, probabilities)


print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {roc:.4f}")

print("\nClassification Report")
print(classification_report(targets, predictions))

print("\nConfusion Matrix")
print(confusion_matrix(targets, predictions))


# ==========================================================
# Save Model
# ==========================================================

torch.save(
    model.state_dict(),
    "models/ft_transformer.pth"
)

print("\nModel Saved Successfully!")
# ==========================================================
# Save Metrics
# ==========================================================

os.makedirs("results", exist_ok=True)

metrics = pd.DataFrame({
    "Model": ["FTTransformer"],
    "Accuracy": [round(accuracy, 4)],
    "Precision": [round(precision, 4)],
    "Recall": [round(recall, 4)],
    "F1 Score": [round(f1, 4)],
    "ROC AUC": [round(roc, 4)]
})

metrics.to_csv(
    "results/ft_transformer_metrics.csv",
    index=False
)

print("Metrics Saved Successfully!")