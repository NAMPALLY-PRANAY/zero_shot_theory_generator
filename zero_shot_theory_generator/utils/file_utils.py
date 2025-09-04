import os, requests

def load_dataset_path(path_or_url):
    if path_or_url.startswith("http"):
        fname = path_or_url.split("/")[-1]
        local_path = os.path.join("downloads", fname)
        os.makedirs("downloads", exist_ok=True)
        r = requests.get(path_or_url)
        with open(local_path, "wb") as f:
            f.write(r.content)
        return local_path
    if os.path.exists(path_or_url):
        return path_or_url
    raise FileNotFoundError(f"{path_or_url} not found")
