# zero_shot_theory_generator/utils/file_utils.py

import os
import requests
import kagglehub
from datasets import load_dataset
import pandas as pd

def load_dataset_path(path_or_url: str) -> str:
    """
    Load dataset from local path, Hugging Face, Kaggle, or direct URLs.

    Args:
        path_or_url (str): Local path, HF URL/slug, Kaggle URL/slug, or direct file URL.

    Returns:
        str: Local path to the dataset (CSV file or extracted folder).
    """

    # 1. Handle local file or folder
    if os.path.exists(path_or_url):
        return os.path.abspath(path_or_url)

    # 2. Handle Hugging Face datasets
    if "huggingface.co" in path_or_url or path_or_url.startswith("hf:"):
        try:
            if path_or_url.startswith("hf:"):
                repo_id = path_or_url.replace("hf:", "")
            else:
                # Example: https://huggingface.co/datasets/<user>/<repo>
                repo_id = "/".join(path_or_url.split("datasets/")[-1].split("/"))

            print(f"[INFO] Downloading Hugging Face dataset: {repo_id}")
            ds_builder = load_dataset(repo_id)

            # Choose split: prefer 'train', else fallback
            split_name = "train" if "train" in ds_builder else list(ds_builder.keys())[0]
            ds = ds_builder[split_name]

            # Convert to Pandas for visualization
            df = ds.to_pandas()

            # Save locally as CSV
            local_dir = os.path.join("downloads", "hf_" + repo_id.replace("/", "_"))
            os.makedirs(local_dir, exist_ok=True)
            csv_path = os.path.join(local_dir, f"{repo_id.replace('/', '_')}.csv")
            df.to_csv(csv_path, index=False)

            return os.path.abspath(csv_path)

        except Exception as e:
            raise RuntimeError(f"Failed to load Hugging Face dataset: {e}")

    # 3. Handle Kaggle datasets via kagglehub
    if path_or_url.startswith("kaggle:") or "kaggle.com/datasets" in path_or_url:
        try:
            if path_or_url.startswith("kaggle:"):
                repo_id = path_or_url.replace("kaggle:", "")
            else:
                # Example: https://www.kaggle.com/datasets/uciml/iris
                repo_id = "/".join(path_or_url.split("datasets/")[-1].split("/"))

            print(f"[INFO] Downloading Kaggle dataset: {repo_id}")
            dataset_path = kagglehub.dataset_download(f"kaggle:{repo_id}")
            return os.path.abspath(dataset_path)

        except Exception as e:
            raise RuntimeError(f"Failed to load Kaggle dataset: {e}")

    # 4. Handle direct file URLs (CSV, JSON, TXT, ZIP)
    if path_or_url.startswith("http"):
        try:
            local_dir = os.path.join("downloads")
            os.makedirs(local_dir, exist_ok=True)

            fname = path_or_url.split("/")[-1]
            if not fname or "." not in fname:
                fname = "dataset_download"

            print(f"[INFO] Downloading file from {path_or_url}")
            r = requests.get(path_or_url, stream=True, timeout=30)
            r.raise_for_status()

            # Guess extension if missing
            if fname == "dataset_download":
                content_type = r.headers.get("Content-Type", "")
                if "csv" in content_type:
                    fname += ".csv"
                elif "json" in content_type:
                    fname += ".json"
                elif "zip" in content_type:
                    fname += ".zip"
                elif "text" in content_type:
                    fname += ".txt"

            local_path = os.path.join(local_dir, fname)

            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            if os.path.getsize(local_path) == 0:
                raise RuntimeError(f"Downloaded file is empty: {local_path}")

            return os.path.abspath(local_path)

        except Exception as e:
            raise RuntimeError(f"Failed to download dataset from {path_or_url}: {e}")

    # 5. Unsupported path
    raise FileNotFoundError(f"Dataset path not found or unsupported: {path_or_url}")
