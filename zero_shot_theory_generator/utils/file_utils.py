import os
import re
import tempfile
import subprocess

def is_kaggle_url(path):
    return "kaggle.com/" in path

def is_hf_slug(path):
    # Accept slugs like "facebook/recycling_the_web"
    return bool(re.match(r"^[\w\-]+/[\w\-]+$", path))

def is_hf_url(path):
    # Accept URLs like "https://huggingface.co/datasets/facebook/recycling_the_web"
    return path.startswith("https://huggingface.co/datasets/")

def download_kaggle(path):
    match = re.search(r"kaggle.com/([^/]+/[^/?]+)", path)
    if not match:
        raise ValueError("Invalid Kaggle URL format.")
    dataset = match.group(1)
    out_dir = tempfile.mkdtemp(prefix="kaggle_")
    subprocess.run([
        "kaggle", "datasets", "download", "-d", dataset, "-p", out_dir, "--unzip"
    ], check=True)
    for f in os.listdir(out_dir):
        fp = os.path.join(out_dir, f)
        if os.path.isfile(fp):
            return fp
    return out_dir

def download_hf(path):
    from datasets import load_dataset
    # Accept both URLs and slugs
    if is_hf_url(path):
        parts = path.rstrip("/").split("/")
        dataset = "/".join(parts[-2:])
    else:
        dataset = path
    out_dir = tempfile.mkdtemp(prefix="hf_")
    ds = load_dataset(dataset)
    split = list(ds.keys())[0]
    csv_path = os.path.join(out_dir, f"{dataset.replace('/', '_')}.csv")
    ds[split].to_csv(csv_path)
    return csv_path

def load_dataset_path(path_or_url):
    # Direct file download for raw file URLs
    if path_or_url.startswith("http") and (
        path_or_url.endswith(".csv")
        or path_or_url.endswith(".json")
        or path_or_url.endswith(".txt")
        or path_or_url.endswith(".zip")
    ):
        import requests
        fname = path_or_url.split("/")[-1]
        local_path = os.path.join("downloads", fname)
        os.makedirs("downloads", exist_ok=True)
        r = requests.get(path_or_url)
        with open(local_path, "wb") as f:
            f.write(r.content)
        return local_path
    # Hugging Face detection (slug or URL)
    if is_hf_url(path_or_url) or is_hf_slug(path_or_url):
        return download_hf(path_or_url)
    # Kaggle detection
    if is_kaggle_url(path_or_url):
        return download_kaggle(path_or_url)
    # Local file
    if os.path.exists(path_or_url):
        return path_or_url
    raise FileNotFoundError(f"Dataset path not found: {path_or_url}")
