import subprocess
import sys
import os
import time

print("=" * 70)
print("           PYROSHIELD-AI MASTER PIPELINE")
print("=" * 70)

scripts = [

    ("Wildfire Prediction", "scripts/predict_wildfire.py"),

    ("Fire Spread Forecast", "scripts/fire_spread_engine.py"),

    ("Decision Support System", "scripts/decision_support.py"),

]

total = len(scripts)

for index, (name, script) in enumerate(scripts, start=1):

    print("\n" + "=" * 70)
    print(f"STEP {index}/{total} : {name}")
    print("=" * 70)

    if not os.path.exists(script):

        print(f"ERROR : {script} not found.")
        sys.exit(1)

    start = time.time()

    result = subprocess.run(
        [sys.executable, script]
    )

    end = time.time()

    if result.returncode != 0:

        print(f"\n{name} FAILED")
        sys.exit(result.returncode)

    print(f"\n{name} Completed Successfully")
    print(f"Execution Time : {end-start:.2f} seconds")

print("\n" + "=" * 70)
print("             PYROSHIELD-AI COMPLETED")
print("=" * 70)

print("\nGenerated Files")

print("----------------------------------------------")

print("✓ results/wildfire_prediction.csv")

print("✓ results/fire_spread_forecast.csv")

print("✓ results/decision_support.csv")

print("----------------------------------------------")

print("\nPipeline Executed Successfully.")