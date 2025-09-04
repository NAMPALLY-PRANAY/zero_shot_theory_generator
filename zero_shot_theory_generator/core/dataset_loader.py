import os, pandas as pd, zipfile, json

def detect_dataset(path, sample_size=100):
    if path.endswith(".csv"):
        df = pd.read_csv(path, nrows=sample_size)
        return {
            "type": "tabular",
            "n_rows": len(df),
            "columns": [{"name": c,
                         "dtype": str(df[c].dtype),
                         "n_unique": int(df[c].nunique()),
                         "missing": float(df[c].isna().mean())}
                         for c in df.columns]
        }

    elif path.endswith(".zip") and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            dirs = [f.split("/")[0] for f in z.namelist() if "/" in f]
            classes = list(set(dirs))
        return {"type": "image_folder", "classes": classes, "n_classes": len(classes)}

    elif path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()[:sample_size]
        return {"type": "text", "sample": lines[:5], "n_lines": len(lines)}

    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        keys = list(data.keys()) if isinstance(data, dict) else []
        return {"type": "json", "keys": keys}

    else:
        raise ValueError(f"Unsupported dataset format: {path}")
