import pandas as pd

# ============================================
# Load Dataset
# ============================================

dataset = pd.read_csv("datasets/final/ml_dataset.csv")

# ============================================
# User Input
# ============================================

latitude = float(input("Enter Latitude : "))
longitude = float(input("Enter Longitude : "))

# ============================================
# Find Nearest Location
# ============================================

dataset["distance"] = (
    (dataset["latitude"] - latitude) ** 2 +
    (dataset["longitude"] - longitude) ** 2
)

nearest = dataset.loc[dataset["distance"].idxmin()]

print("\nNearest Location Found")
print(nearest[["latitude", "longitude"]])