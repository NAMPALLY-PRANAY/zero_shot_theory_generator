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
        # Try to parse as JSONL
        try:
            json_objs = [json.loads(line) for line in lines]
            if isinstance(json_objs[0], dict):
                keys = list(json_objs[0].keys())
                return {"type": "jsonl", "keys": keys, "n_lines": len(lines)}
        except Exception:
            pass
        return {"type": "text", "sample": lines[:5], "n_lines": len(lines)}

    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            keys = list(data.keys())
            return {"type": "json", "keys": keys}
        elif isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                keys = list(data[0].keys())
                return {"type": "json_list", "keys": keys, "n_items": len(data)}
            else:
                return {"type": "text_list", "n_items": len(data), "sample": data[:5]}
        else:
            return {"type": "json_unknown", "sample": str(type(data))}

    else:
        # Try to infer tabular from extensionless or unknown files
        try:
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
        except Exception:
            pass
        raise ValueError(f"Unsupported dataset format: {path}")
