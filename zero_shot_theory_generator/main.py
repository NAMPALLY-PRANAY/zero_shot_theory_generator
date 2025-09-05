import argparse
import os
import sys
import time
from zero_shot_theory_generator.config.settings import OUTPUT_DIR
from zero_shot_theory_generator.utils.file_utils import load_dataset_path
from zero_shot_theory_generator.core.dataset_loader import detect_dataset
from zero_shot_theory_generator.core.task_inference import infer_task
from zero_shot_theory_generator.core.pipeline_suggester import suggest_pipeline
from zero_shot_theory_generator.core.theory_generator import generate_theory
from zero_shot_theory_generator.core.explainability import explain_pipeline
from zero_shot_theory_generator.utils.logger import log_output

def paradigm_and_strategy(task, meta):
    paradigm = "Unknown"
    strategy = []
    if task["task"] in ["classification", "regression", "image_classification", "text_classification"]:
        paradigm = "Supervised Learning"
        n_rows = meta.get("n_rows", 100)
        batch_size = min(64, max(8, n_rows // 10))
        epochs = 10 if n_rows < 1000 else 50
        split = "80% train / 20% test" if n_rows > 50 else "Leave-one-out"
        cross_val = "5-fold" if n_rows > 200 else "3-fold"
        optimizer = "Adam" if task["task"] in ["classification", "image_classification", "text_classification"] else "SGD"
        early_stopping = "Yes" if n_rows > 100 else "No"
        strategy = [
            f"Split: {split}",
            f"Cross-validation: {cross_val}",
            f"Epochs: {epochs}",
            f"Batch size: {batch_size}",
            f"Optimizer: {optimizer}",
            f"Early stopping: {early_stopping}"
        ]
    elif task["task"] == "unsupervised":
        paradigm = "Unsupervised Learning"
        n_cols = len(meta.get("columns", []))
        clustering = "K-means" if n_cols < 10 else "DBSCAN"
        dim_red = "PCA" if n_cols < 20 else "t-SNE"
        association = "Apriori" if n_cols > 2 else "None"
        strategy = [
            f"Clustering: {clustering}",
            f"Dimensionality Reduction: {dim_red}",
            f"Association Rule Mining: {association}"
        ]
    else:
        paradigm = "Unknown/Other"
        strategy = ["Custom analysis required"]
    return paradigm, strategy

def format_output(meta, task, pipeline, theory):
    dataset_md = f"## 📊 Dataset\n"
    if meta.get("type") == "image_folder":
        dataset_md += f"**Type:** Image Folder\n**Classes:** {meta.get('classes', [])}\n"
    elif meta.get("type") == "tabular":
        cols = meta.get("columns", [])
        dataset_md += f"**Type:** Tabular\n**Columns:** {', '.join([c['name'] for c in cols])}\n"
    elif meta.get("type") == "text":
        dataset_md += f"**Type:** Text\n**Sample:**\n```\n{''.join(meta.get('sample', []))}\n```\n"
    elif meta.get("type", "").startswith("json"):
        dataset_md += f"**Type:** JSON\n**Keys:** {meta.get('keys', [])}\n"
    else:
        dataset_md += f"**Type:** {meta.get('type')}\n"

    task_md = f"## 🎯 Task\n"
    for k, v in task.items():
        task_md += f"- **{k.capitalize()}**: {v}\n"

    pipeline_md = "## 🛠️ Pipeline Suggestion\n"
    if isinstance(pipeline, dict):
        for k, v in pipeline.items():
            pipeline_md += f"- **{k.capitalize()}**: {v}\n"
    else:
        pipeline_md += f"{pipeline}\n"

    paradigm, strategy = paradigm_and_strategy(task, meta)
    strategy_md = f"## 🌍 ML Paradigm & Training Strategy\n**Paradigm:** {paradigm}\n**Recommended Strategy:**\n"
    for s in strategy:
        strategy_md += f"- {s}\n"

    explain_md = "## 🔍 Explainability\n"
    model_name = pipeline.get("model") if isinstance(pipeline, dict) else None
    explain_md += explain_pipeline(model_name) + "\n"

    theory_md = "## 🧪 Scientific Theory Insights\n"
    rules = theory.get("rules", [])
    if rules:
        for r in rules:
            theory_md += f"- {r}\n"
    llm = theory.get("llm", "")
    if llm:
        theory_md += f"\n**LLM Insights:**\n{llm}\n"

    return f"# 🧠 Zero-Shot AI Theory Generator\n\n{dataset_md}\n{task_md}\n{pipeline_md}\n{strategy_md}\n{explain_md}\n{theory_md}"

def print_live(text, delay=0.01):
    for line in text.splitlines():
        print(line)
        time.sleep(delay)

def analyze(path_or_file):
    try:
        if hasattr(path_or_file, "name"):
            local_path = path_or_file.name
        else:
            local_path = path_or_file
        os.makedirs(os.path.join(OUTPUT_DIR, "reports"), exist_ok=True)
        dataset_path = load_dataset_path(local_path)
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
        status_msg = f"Report saved to {os.path.join(OUTPUT_DIR, 'reports')}/"
        if "GOOGLE_API_KEY not set" in str(theory.get("llm", "")):
            status_msg += " [Gemini API key missing!]"
        output_md = format_output(meta, task, pipeline, theory)
        return output_md, status_msg
    except Exception as e:
        return f"**Error:** {str(e)}", f"Error: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Zero-Shot Theory Generator")
    parser.add_argument("--mode", type=str, choices=["file", "url", "both"], default="file",
                        help="Input mode: file, url, or both (interactive)")
    parser.add_argument("--path", type=str, help="Local dataset path")
    parser.add_argument("--url", type=str, help="Dataset URL")
    args = parser.parse_args()

    # Switch-case for input mode
    match args.mode:
        case "file":
            if not args.path:
                print("Please provide --path for file mode.")
                sys.exit(1)
            input_source = args.path
        case "url":
            if not args.url:
                print("Please provide --url for url mode.")
                sys.exit(1)
            input_source = args.url
        case "both":
            print("Interactive mode: Enter file path or URL.")
            inp = input("File path (leave blank to use URL): ").strip()
            if inp:
                input_source = inp
            else:
                inp_url = input("URL: ").strip()
                if not inp_url:
                    print("No input provided.")
                    sys.exit(1)
                input_source = inp_url
        case _:
            print("Invalid mode.")
            sys.exit(1)

    print("\nAnalyzing dataset... Please wait.\n")
    output_md, status_msg = analyze(input_source)
    print_live(output_md, delay=0.01)
    print(f"\n{status_msg}")

if __name__ == "__main__":
    main()
