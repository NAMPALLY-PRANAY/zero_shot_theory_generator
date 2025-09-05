# zero_shot_theory_generator/utils/file_utils.py

import os
import re
import requests
import pandas as pd
from typing import Optional
from datasets import load_dataset as hf_load_dataset
import kagglehub


def extract_kaggle_slug(path_or_url: str) -> Optional[str]:
    """Extract Kaggle dataset slug like 'owner/dataset'."""
    m = re.search(r"(?:kaggle:|kaggle\.com/(?:datasets/)?)([\w\-]+/[\w\-]+)", path_or_url)
    if m:
        return m.group(1)
    m2 = re.match(r"^[\w\-]+/[\w\-]+$", path_or_url.strip())
    if m2:
        return path_or_url.strip()
    return None


def save_dataframe(df, out_dir, fname="data.csv"):
    """Save DataFrame safely to CSV and return path."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, fname)
    df.to_csv(out_path, index=False)
    return os.path.abspath(out_path)


def load_dataset_path(path_or_url: str) -> str:
    """
    Detect and normalize dataset input into a local usable file/folder:
      - Local path
      - Hugging Face Hub
      - Kaggle datasets
      - Direct file URL
    Returns local filesystem path (CSV, TXT, JSON, folder).
    """

    # 1. Local file/folder
    if os.path.exists(path_or_url):
        return os.path.abspath(path_or_url)

    # 2. Hugging Face Hub
    hf_match = None
    if path_or_url.startswith("hf:"):
        hf_match = path_or_url[3:]
    else:
        m = re.match(r".*huggingface\.co/datasets/([^/]+/[^/:]+)(?::([^/]+))?", path_or_url)
        if m:
            repo_id, config = m.group(1), m.group(2)
            hf_match = f"{repo_id}:{config}" if config else repo_id

    if hf_match:
        repo_id, config = (hf_match.split(":", 1) + [None])[:2]
        print(f"[INFO] Loading Hugging Face dataset → {repo_id}, config={config}")
        try:
            ds = hf_load_dataset(repo_id, config) if config else hf_load_dataset(repo_id)
            local_dir = os.path.join("downloads", "huggingface", repo_id.replace("/", "_"))
            os.makedirs(local_dir, exist_ok=True)

            # Handle split
            data = ds[list(ds.keys())[0]] if isinstance(ds, dict) else ds

            # Try saving as CSV
            try:
                df = data.to_pandas()
                return save_dataframe(df, local_dir, "dataset.csv")
            except Exception:
                save_path = os.path.join(local_dir, "dataset.parquet")
                data.to_parquet(save_path)
                return os.path.abspath(save_path)

        except Exception as e:
            raise RuntimeError(f"Failed to load Hugging Face dataset: {e}")

    # 3. Kaggle datasets
    kaggle_slug = extract_kaggle_slug(path_or_url)
    if kaggle_slug:
        print(f"[INFO] Downloading Kaggle dataset: {kaggle_slug}")
        try:
            # ✅ Download Kaggle dataset to a local folder
            dataset_path = kagglehub.dataset_download(kaggle_slug)
        except Exception as e:
            raise RuntimeError(f"Failed to load Kaggle dataset: {e}")

        # Look for CSV files inside the dataset folder
        csv_files = []
        for root, _, files in os.walk(dataset_path):
            for f in files:
                if f.endswith(".csv"):
                    csv_files.append(os.path.join(root, f))

        if not csv_files:
            raise RuntimeError("No CSV files found in Kaggle dataset")

        # If multiple CSVs, return the folder; else return single CSV path
        if len(csv_files) == 1:
            return os.path.abspath(csv_files[0])
        else:
            return os.path.abspath(dataset_path)

    # 4. Direct file URLs
    if path_or_url.startswith("http"):
        local_dir = os.path.join("downloads", "raw")
        os.makedirs(local_dir, exist_ok=True)
        fname = os.path.basename(path_or_url).split("?")[0] or "dataset_download"
        local_path = os.path.join(local_dir, fname)
        print(f"[INFO] Downloading dataset file → {path_or_url}")
        try:
            r = requests.get(path_or_url, stream=True, timeout=30)
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)
            if os.path.getsize(local_path) == 0:
                raise RuntimeError("Downloaded file is empty")
            return os.path.abspath(local_path)
        except Exception as e:
            raise RuntimeError(f"Failed to download dataset file: {e}")

    # 5. Unsupported
    raise FileNotFoundError(f"Unsupported dataset path or URL: {path_or_url}")
