# 🧠 Zero-Shot AI Theory Generator

## 🚀 Project Summary

**Zero-Shot Theory Generator** is an automated AI framework that analyzes arbitrary datasets and generates machine learning task predictions, pipeline suggestions, and scientific research-style insights — all without prior labels or manual configuration.

---

## 📥 Input

- **Flexible Input Modes:**  
  - Local file path  
  - Direct URL  
  - Hugging Face datasets  
  - Kaggle datasets

- **Automatic Handling:**  
  - Detects dataset source  
  - Downloads and preprocesses into a uniform format

---

## 🧩 Core Components

- **Dataset Loader:**  
  Robust extraction from multiple sources (local, Hugging Face Hub, Kaggle, raw URLs).

- **Dataset Detector:**  
  Inspects structure (tabular, text, image, JSON) and metadata (rows, columns, uniqueness, sample values).

- **Task Inference:**  
  Zero-shot inference of dataset type → classification, regression, clustering, text classification, etc.

- **Pipeline Suggester:**  
  Proposes preprocessing, models, loss functions, and evaluation metrics based on inferred task.

- **Theory Generator:**  
  Produces scientific insights and research-style interpretations about dataset quality, limitations, and possible improvements using an LLM.

- **Explainability Module:**  
  Adds interpretability suggestions (e.g., LIME, SHAP, attention visualization).

---

## 📤 Output

Generates a structured research-style report containing:

- **📊 Dataset Summary:** Type, columns, samples
- **🎯 Task Prediction:** (e.g., classification, regression, clustering,time-series)
- **🛠️ Pipeline Suggestion:** Preprocessing, models, metrics
- **🌍 ML Paradigm & Training Strategy:** Supervised/unsupervised, split ratios, cross-validation, optimizers
- **🔍 Explainability Recommendations**
- **🧪 Scientific Theory Insights:** Data limitations, preprocessing needs, model suitability, improvement directions

---

## 📝 Example: Dengue Prediction Dataset

- **Dataset:** Weather + Dengue case counts
- **Task Predicted:** Unsupervised (clustering)
- **Pipeline Suggested:** StandardScaler + DBSCAN + t-SNE
- **Scientific Insights:**  
  - Small dataset size  
  - Noisy features  
  - Limited label variety  
  - Importance of preprocessing and feature engineering  
  - Model trade-offs

---

## 💡 Applications

- **Automated ML Research Assistance:**  
  Quickly hypothesize tasks and pipelines for unknown datasets.

- **Educational Tool:**  
  Teaching ML pipeline design and dataset analysis.

- **Preliminary Analysis for Researchers:**  
  Generates interpretable starting points for deeper model development.

---

## ⚠️ Limitations

- Zero-shot predictions may misclassify task type if labels are ambiguous or missing.
- Suggested pipelines are heuristics, not guaranteed optimal models.
- Scientific insights depend on LLM interpretation, which may vary in accuracy.

---

## 🛠️ Tools & Technologies Used

- **Python** (core language)
- **PyTorch / TensorFlow** (for model suggestions)
- **Hugging Face Datasets** (dataset loading)
- **Kaggle API** (dataset access)
- **LLMs** (for scientific theory generation)
- **Explainability Libraries:** LIME, SHAP, etc.

---

## ⚡ In One Line

> An AI-driven zero-shot framework that turns any dataset into a research-style analysis report with task inference, pipeline suggestions, and scientific insights — without requiring labels or manual configuration.

---

## 🏁 Setup

```bash
git clone https://github.com/NAMPALLY-PRANAY/zero_shot_theory_generator.git
cd zero_shot_theory_generator
pip install -r requirements.txt
```

**Before running the application, create a `.env` file in the project root with your Google Gemini API key:**

```bash
echo GOOGLE_API_KEY=<APIKEY> > .env
```
*(Replace `<APIKEY>` with your actual API key, no quotes or spaces.)*

---

## ▶️ Running

- **Web GUI:**  
  ```bash
  gradio app.py
  ```
- **Command Line Interface:**  
  ```bash
  python main.py
  ```
