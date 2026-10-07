import pandas as pd

fire = pd.read_csv("datasets/final/wildfire_features.csv")
nonfire = pd.read_csv("datasets/final/nonfire_features.csv")

# Add target labels
fire["fire"] = 1

# Ensure non-fire label exists
if "fire" not in nonfire.columns:
    nonfire["fire"] = 0

dataset = pd.concat([fire, nonfire], ignore_index=True)

# Shuffle the dataset
dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

dataset.to_csv(
    "datasets/final/training_dataset.csv",
    index=False
)

print("=" * 50)
print("Training Dataset Created Successfully")
print("=" * 50)
print("Fire samples     :", len(fire))
print("Non-fire samples :", len(nonfire))
print("Total samples    :", len(dataset))