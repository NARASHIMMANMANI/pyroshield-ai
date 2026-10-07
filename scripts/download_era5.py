import pandas as pd

df = pd.read_csv(
    "datasets/processed/wildfire_events_2020_2024.csv",
    nrows=5
)

print(df["satellite"].value_counts())
print(df["instrument"].value_counts())