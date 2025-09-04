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
from zero_shot_theory_generator.utils.logger import log_output

def analyze(path):
    try:
        dataset_path = load_dataset_path(path)
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
        return str(meta), str(task), str(pipeline), str(theory), "Report saved to outputs/reports/"
    except Exception as e:
        return "", "", "", "", f"Error: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Zero-Shot Theory Generator")
    with gr.Row():
        path = gr.Textbox(label="Dataset Path or URL", interactive=True)
    with gr.Row():
        meta = gr.Textbox(label="Metadata", interactive=False)
        task = gr.Textbox(label="Inferred Task", interactive=False)
    with gr.Row():
        pipeline = gr.Textbox(label="Pipeline Suggestion", interactive=False)
        theory = gr.Textbox(label="Generated Theory", interactive=False)
    msg = gr.Textbox(label="Status", interactive=False)
    btn = gr.Button("Analyze")
    btn.click(analyze, inputs=path, outputs=[meta, task, pipeline, theory, msg])

if __name__ == "__main__":
    demo.launch()
