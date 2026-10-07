import os
import pandas as pd

print("=" * 60)
print("PYROSHIELD-AI DECISION SUPPORT SYSTEM")
print("=" * 60)

# =====================================================
# Load Fire Spread Forecast
# =====================================================

df = pd.read_csv("results/fire_spread_forecast.csv")

print("Fire Spread Forecast Loaded Successfully")
print("Shape :", df.shape)

# =====================================================
# Generate Recommendations
# =====================================================

alert_level = []
priority = []
monitoring = []
response = []
evacuation = []

for _, row in df.iterrows():

    probability = row["fire_probability"]
    risk = row["risk_level"]
    distance = row["spread_distance_km"]

    if risk == "HIGH":

        alert_level.append("RED")
        priority.append("CRITICAL")
        monitoring.append("Every 15 Minutes")

        response.append(
            "Deploy firefighters immediately; Dispatch drones; Notify Forest Department"
        )

        evacuation.append(
            "Prepare evacuation if settlements are within spread distance."
        )

    elif risk == "MEDIUM":

        alert_level.append("ORANGE")
        priority.append("HIGH")
        monitoring.append("Every 30 Minutes")

        response.append(
            "Deploy surveillance teams; Increase monitoring"
        )

        evacuation.append(
            "Keep evacuation teams on standby."
        )

    else:

        alert_level.append("GREEN")
        priority.append("LOW")
        monitoring.append("Every 2 Hours")

        response.append(
            "Routine monitoring only"
        )

        evacuation.append(
            "No evacuation required."
        )

# =====================================================
# Create Final Decision Table
# =====================================================

decision = df.copy()

decision["alert_level"] = alert_level
decision["priority"] = priority
decision["monitoring_frequency"] = monitoring
decision["recommended_action"] = response
decision["evacuation_plan"] = evacuation

# =====================================================
# Save Results
# =====================================================

os.makedirs("results", exist_ok=True)

decision.to_csv(
    "results/decision_support.csv",
    index=False
)

print("\nDecision Support Report Saved Successfully!")

print("\nLocation : results/decision_support.csv")

print("\nPreview\n")

print(
    decision[
        [
            "latitude",
            "longitude",
            "fire_probability",
            "risk_level",
            "alert_level",
            "priority",
            "monitoring_frequency",
        ]
    ].head()
)

# =====================================================
# Summary
# =====================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print("\nAlert Levels")
print(decision["alert_level"].value_counts())

print("\nPriorities")
print(decision["priority"].value_counts())

print("\nRisk Levels")
print(decision["risk_level"].value_counts())