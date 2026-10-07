import pandas as pd

print("=" * 60)
print("CONFORMAL EVALUATION")
print("=" * 60)

df = pd.read_csv(
    "results/uncertainty/conformal_predictions.csv"
)

print()

print("Total Predictions :", len(df))

print("Reliable Predictions :", df["Reliable"].sum())

print("Reliability (%) :",
      round(df["Reliable"].mean()*100,2))