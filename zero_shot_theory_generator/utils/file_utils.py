import os
import re
import tempfile
import subprocess
import time

def is_kaggle_url(path):
    return "kaggle.com/datasets/" in path

def is_hf_slug(path):
    return bool(re.match(r"^[\w\-]+/[\w\-]+$", path))

def is_hf_url(path):
    return path.startswith("https://huggingface.co/datasets/")

def download_kaggle(path):
    match = re.search(r"kaggle.com/([^/]+/[^/?]+)", path)
    if not match:
        raise ValueError("Invalid Kaggle URL format.")
    dataset = match.group(1)
    out_dir = tempfile.mkdtemp(prefix="kaggle_")
    try:
        subprocess.run([
            "kaggle", "datasets", "download", "-d", dataset, "-p", out_dir, "--unzip"
        ], check=True)
    except Exception as e:
        raise RuntimeError(f"Kaggle download failed: {e}")
    for f in os.listdir(out_dir):
        fp = os.path.join(out_dir, f)
        if os.path.isfile(fp):
            return fp
    return out_dir

def download_hf(path, batch_size=100):
    from datasets import load_dataset
    import pandas as pd

    if is_hf_url(path):
        parts = path.rstrip("/").split("/")
        dataset = "/".join(parts[-2:])
    else:
        dataset = path
    out_dir = tempfile.mkdtemp(prefix="hf_")
    ds = load_dataset(dataset, split="train", streaming=True)
    csv_path = os.path.join(out_dir, f"{dataset.replace('/', '_')}.csv")

    # Stream and save in batches to avoid rate limiting
    batch = []
    total = 0
    try:
        for i, example in enumerate(ds):
            batch.append(example)
            if len(batch) >= batch_size:
                df = pd.DataFrame(batch)
                if total == 0:
                    df.to_csv(csv_path, index=False, mode='w')
                else:
                    df.to_csv(csv_path, index=False, mode='a', header=False)
                total += len(batch)
                batch = []
                time.sleep(1)  # Small delay to avoid rate limit
        # Save any remaining examples
        if batch:
            df = pd.DataFrame(batch)
            if total == 0:
                df.to_csv(csv_path, index=False, mode='w')
            else:
                df.to_csv(csv_path, index=False, mode='a', header=False)
    except Exception as e:
        raise RuntimeError(f"Hugging Face batch download failed: {e}")
    return csv_path

def robust_request(url, retries=5, backoff=2):
    import requests
    for attempt in range(retries):
        try:
            r = requests.get(url)
            r.raise_for_status()
            return r
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1 and (hasattr(e.response, 'status_code') and e.response.status_code == 429):
                time.sleep(backoff ** attempt)
            else:
                raise

def load_dataset_path(path_or_url):
    # Direct file download for raw file URLs
    if path_or_url.startswith("http") and (
        path_or_url.endswith(".csv")
        or path_or_url.endswith(".json")
        or path_or_url.endswith(".txt")
        or path_or_url.endswith(".zip")
    ):
        fname = path_or_url.split("/")[-1]
        downloads_dir = os.path.abspath("downloads")
        os.makedirs(downloads_dir, exist_ok=True)
        local_path = os.path.join(downloads_dir, fname)
        try:
            r = robust_request(path_or_url)
            with open(local_path, "wb") as f:
                f.write(r.content)
        except Exception as e:
            raise RuntimeError(f"Failed to download file: {e}")
        return local_path
    if is_hf_url(path_or_url) or is_hf_slug(path_or_url):
        # Use batch download for Hugging Face datasets
        return download_hf(path_or_url, batch_size=100)
    if is_kaggle_url(path_or_url):
        return download_kaggle(path_or_url)
    if os.path.exists(path_or_url):
        return path_or_url
    raise FileNotFoundError(f"Dataset path not found: {path_or_url}")
