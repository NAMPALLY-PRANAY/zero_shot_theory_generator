from setuptools import setup, find_packages

setup(
    name="zero_shot_theory_generator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy", 
        "scikit-learn",
        "torch",
        "torchvision",
        "transformers",
        "datasets",
        "gradio",
        "shap",
        "lime",
        "python-dotenv",
        "requests",
        "openai",
    ],
    python_requires=">=3.8",
)