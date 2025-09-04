import argparse
import os
from zero_shot_theory_generator.config.settings import OUTPUT_DIR
from zero_shot_theory_generator.utils.file_utils import load_dataset_path
from zero_shot_theory_generator.core.dataset_loader import detect_dataset
from zero_shot_theory_generator.core.task_inference import infer_task
from zero_shot_theory_generator.core.pipeline_suggester import suggest_pipeline
from zero_shot_theory_generator.core.theory_generator import generate_theory
from zero_shot_theory_generator.utils.logger import log_output

def main():
    parser = argparse.ArgumentParser(description="Zero-Shot Theory Generator")
    parser.add_argument("--path", type=str, required=True,
                        help="Local dataset path or URL")
    args = parser.parse_args()

    dataset_path = load_dataset_path(args.path)

    meta = detect_dataset(dataset_path)
    task = infer_task(meta)
    pipeline = suggest_pipeline(task, meta)
    theory = generate_theory(meta, task, pipeline)

    report = {
        "metadata": meta,
        "task": task,
        "pipeline": pipeline,
        "theory": theory
    }

    os.makedirs(os.path.join(OUTPUT_DIR, "reports"), exist_ok=True)
    log_output(report)
    print(f"\n=== Report saved to {os.path.join(OUTPUT_DIR, 'reports')}/ ===")

if __name__ == "__main__":
    main()
