import os, json, time
from zero_shot_theory_generator.config.settings import OUTPUT_DIR

def log_output(report):
    reports_dir = os.path.join(OUTPUT_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    filename = os.path.join(reports_dir, f"report_{int(time.time())}.json")
    try:
        with open(filename, "w") as f:
            json.dump(report, f, indent=4)
        print(f"Report saved: {filename}")
    except Exception as e:
        print(f"Failed to save report: {e}")
