import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gradio as gr
from zero_shot_theory_generator.utils.file_utils import load_dataset_path
from zero_shot_theory_generator.core.dataset_loader import detect_dataset
from zero_shot_theory_generator.core.task_inference import infer_task
from zero_shot_theory_generator.core.pipeline_suggester import suggest_pipeline
from zero_shot_theory_generator.core.theory_generator import generate_theory

def analyze(path):
    dataset_path = load_dataset_path(path)
    meta = detect_dataset(dataset_path)
    task = infer_task(meta)
    pipeline = suggest_pipeline(task, meta)
    theory = generate_theory(meta, task, pipeline)
    return str(meta), str(task), str(pipeline), str(theory)

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
    btn = gr.Button("Analyze")
    btn.click(analyze, inputs=path, outputs=[meta, task, pipeline, theory])

if __name__ == "__main__":
    demo.launch()
