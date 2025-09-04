import argparse
from utils.file_utils import load_dataset_path
from core.dataset_loader import detect_dataset
from core.task_inference import infer_task
from core.pipeline_suggester import suggest_pipeline
from core.theory_generator import generate_theory
from utils.logger import log_output

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

    log_output(report)
    print("\n=== Report saved to outputs/reports/ ===")

if __name__ == "__main__":
    main()
