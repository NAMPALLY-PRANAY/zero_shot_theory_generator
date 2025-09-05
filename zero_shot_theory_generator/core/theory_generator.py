from zero_shot_theory_generator.config.settings import GEMINI_API_KEY
import google.generativeai as genai

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def generate_theory(meta, task, pipeline):
    base_theories = []

    # Add task-specific base theories
    if task.get("task") == "classification":
        cols = meta.get("columns", [])
        for c in cols:
            if c["name"].lower() in ("target", "label"):
                if c["n_unique"] > 2:
                    base_theories.append("Multi-class tasks benefit from deeper architectures with residual connections.")
                else:
                    base_theories.append("Binary tasks align well with logistic loss functions.")
                    
    elif task.get("task") == "time_series_forecasting":
        base_theories.append("Time series forecasting benefits from models that capture temporal dependencies.")
        base_theories.append("Seasonal patterns require specialized decomposition techniques.")
        base_theories.append("Feature engineering (lags, rolling statistics) often improves forecasting accuracy.")
    
    # Generate detailed theory with LLM if API key is available
    if not GEMINI_API_KEY:
        return {"rules": base_theories, "llm": "Gemini LLM error: GOOGLE_API_KEY not set. Please set it in your .env file."}

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Customize prompt based on task type
        task_type = task.get("task", "unknown")
        
        if task_type == "time_series_forecasting":
            prompt = (
                "You are an AI that generates ML theory insights for time series forecasting.\n"
                f"Dataset metadata: {meta}\nTask: {task}\nPipeline: {pipeline}\n"
                "Suggest 3 scientific insights about time series forecasting for this data. "
                "Consider seasonality, trend analysis, and forecasting horizons. "
                "Format your answer in Markdown with clear sections."
            )
        else:
            prompt = (
                "You are an AI that generates ML theory insights.\n"
                f"Dataset metadata: {meta}\nTask: {task}\nPipeline: {pipeline}\n"
                "Suggest 3 scientific insights. Format your answer in Markdown with clear sections."
            )
            
        response = model.generate_content(prompt)
        llm_theory = response.text if hasattr(response, "text") else str(response)
        llm_theory = llm_theory.replace("\n\n", "\n").strip()
        return {"rules": base_theories, "llm": llm_theory}
    except Exception as e:
        err_msg = (
            f"Gemini LLM error: {e}\n"
            "Check that GOOGLE_API_KEY is set in your environment or .env file."
        )
        return {"rules": base_theories, "llm": err_msg}
