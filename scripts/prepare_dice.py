import joblib
import pandas as pd
import dice_ml

print("=" * 60)
print("PREPARING DiCE")
print("=" * 60)

# Load dataset
df = pd.read_csv("datasets/final/ml_dataset.csv")

# Load trained Random Forest model
model = joblib.load("models/random_forest.pkl")

# Create DiCE Data object
data = dice_ml.Data(
    dataframe=df,
    continuous_features=[
        c for c in df.columns if c != "fire"
    ],
    outcome_name="fire"
)

# Create DiCE Model object
dice_model = dice_ml.Model(
    model=model,
    backend="sklearn"
)

# Save objects
joblib.dump(data, "models/dice_data.pkl")
joblib.dump(dice_model, "models/dice_model.pkl")

print()
print("DiCE Objects Saved Successfully!")