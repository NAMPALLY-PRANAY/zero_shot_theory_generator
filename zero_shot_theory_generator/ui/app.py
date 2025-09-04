import sys, os
# Add the project root to sys.path for proper imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import gradio as gr
from zero_shot_theory_generator.utils.file_utils import load_dataset_path
from zero_shot_theory_generator.core.dataset_loader import detect_dataset
from zero_shot_theory_generator.core.task_inference import infer_task
from zero_shot_theory_generator.core.pipeline_suggester import suggest_pipeline
from zero_shot_theory_generator.core.theory_generator import generate_theory
from zero_shot_theory_generator.core.explainability import explain_pipeline
from zero_shot_theory_generator.utils.logger import log_output
from zero_shot_theory_generator.config.settings import OUTPUT_DIR

def paradigm_and_strategy(task, meta):
    paradigm = "Unknown"
    strategy = []
    # Supervised learning
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
    # Unsupervised learning
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
    # Dataset summary
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

    # Task summary
    task_md = f"## 🎯 Task\n"
    for k, v in task.items():
        task_md += f"- **{k.capitalize()}**: {v}\n"

    # Pipeline summary
    pipeline_md = "## 🛠️ Pipeline Suggestion\n"
    if isinstance(pipeline, dict):
        for k, v in pipeline.items():
            pipeline_md += f"- **{k.capitalize()}**: {v}\n"
    else:
        pipeline_md += f"{pipeline}\n"

    # Paradigm and Strategy summary
    paradigm, strategy = paradigm_and_strategy(task, meta)
    strategy_md = f"## 🌍 ML Paradigm & Training Strategy\n**Paradigm:** {paradigm}\n**Recommended Strategy:**\n"
    for s in strategy:
        strategy_md += f"- {s}\n"

    # Explainability summary
    explain_md = "## 🔍 Explainability\n"
    model_name = pipeline.get("model") if isinstance(pipeline, dict) else None
    explain_md += explain_pipeline(model_name) + "\n"

    # Theory summary
    theory_md = "## 🧪 Scientific Theory Insights\n"
    rules = theory.get("rules", [])
    if rules:
        for r in rules:
            theory_md += f"- {r}\n"
    llm = theory.get("llm", "")
    if llm:
        theory_md += f"\n**LLM Insights:**\n{llm}\n"

    return f"# 🧠 Zero-Shot AI Theory Generator\n\n{dataset_md}\n{task_md}\n{pipeline_md}\n{strategy_md}\n{explain_md}\n{theory_md}"

def analyze(path_or_file):
    try:
        # Handle file upload or URL
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

with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Zero-Shot Theory Generator")
    with gr.Row():
        file_input = gr.File(label="Upload Dataset (CSV, ZIP, TXT, JSON)", interactive=True)
        url_input = gr.Textbox(label="Or Enter Dataset URL", interactive=True)
    with gr.Row():
        output_md = gr.Markdown()
    msg = gr.Textbox(label="Status", interactive=False)
    btn = gr.Button("Analyze")
    def analyze_wrapper(file, url):
        # Prefer file if uploaded, else use URL
        return analyze(file if file else url)
    btn.click(analyze_wrapper, inputs=[file_input, url_input], outputs=[output_md, msg])
    iterate_btn = gr.Button("Iterate")
    iterate_btn.click(analyze_wrapper, inputs=[file_input, url_input], outputs=[output_md, msg])

if __name__ == "__main__":
    demo.launch()
