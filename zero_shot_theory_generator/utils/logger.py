import os, json, time
from zero_shot_theory_generator.config.settings import OUTPUT_DIR

def log_output(report):
    os.makedirs(f"{OUTPUT_DIR}/reports", exist_ok=True)
    filename = f"{OUTPUT_DIR}/reports/report_{int(time.time())}.json"
    with open(filename, "w") as f:
        json.dump(report, f, indent=4)
    print(f"Report saved: {filename}")
