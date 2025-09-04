import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".venv", ".env"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OUTPUT_DIR = os.path.join(os.getcwd(), "outputs")
