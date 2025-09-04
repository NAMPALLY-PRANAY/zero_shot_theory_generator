from config.settings import OPENAI_API_KEY
import openai

openai.api_key = OPENAI_API_KEY

def generate_theory(meta, task, pipeline):
    base_theories = []

    if task.get("task") == "classification":
        cols = meta.get("columns", [])
        for c in cols:
            if c["name"].lower() in ("target", "label"):
                if c["n_unique"] > 2:
                    base_theories.append("Multi-class tasks benefit from deeper architectures with residual connections.")
                else:
                    base_theories.append("Binary tasks align well with logistic loss functions.")

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system", "content":"You are an AI that generates ML theory insights."},
                {"role":"user", "content":f"Dataset metadata: {meta}\nTask: {task}\nPipeline: {pipeline}\nSuggest 3 scientific insights."}
            ]
        )
        llm_theory = response["choices"][0]["message"]["content"]
        return {"rules": base_theories, "llm": llm_theory}
    except Exception as e:
        return {"rules": base_theories, "llm": f"LLM error: {e}"}
