import pandas as pd
from pathlib import Path

print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

# =====================================================
# Load Metrics Files
# =====================================================

rf = pd.read_csv("results/random_forest_metrics.csv")
xgb = pd.read_csv("results/xgboost_metrics.csv")
cat = pd.read_csv("results/catboost_metrics.csv")
ft = pd.read_csv("results/ft_transformer_metrics.csv")
kan = pd.read_csv("results/kan_metrics.csv")

# =====================================================
# Merge All Results
# =====================================================

comparison = pd.concat(
    [rf, xgb, cat, ft, kan],
    ignore_index=True
)

# =====================================================
# Sort by Accuracy
# =====================================================

comparison = comparison.sort_values(
    by="Accuracy",
    ascending=False
).reset_index(drop=True)

# =====================================================
# Save Results
# =====================================================

Path("results").mkdir(exist_ok=True)

comparison.to_csv(
    "results/model_comparison.csv",
    index=False
)

# =====================================================
# Display Results
# =====================================================

print(comparison)

print("\nComparison Saved Successfully!")
print("Location : results/model_comparison.csv")