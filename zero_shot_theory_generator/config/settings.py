import os
from dotenv import load_dotenv

project_root = os.path.dirname(os.path.dirname(__file__))

# Try loading .env from .venv/.env and project root
env_paths = [
    os.path.join(project_root, ".venv", ".env"),
    os.path.join(project_root, ".env")
]
for env_path in env_paths:
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path)

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    # Static fallback for Gemini API key
    GEMINI_API_KEY = ""
OUTPUT_DIR = os.path.join(project_root, "ui", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
