def infer_task(meta):
    if meta["type"] == "image_folder":
        return {"task": "image_classification", "confidence": 0.95}

    # Detect tabular with image columns for CNN optimization
    if meta["type"] == "tabular":
        image_cols = [c for c in meta["columns"] if c["name"].lower() in ("image_path", "image_url", "img", "image")]
        if image_cols:
            return {"task": "image_classification", "input": image_cols[0]["name"], "confidence": 0.9}
        for c in meta["columns"]:
            if c["name"].lower() in ("target", "label", "y"):
                if c["n_unique"] <= 50:
                    return {"task": "classification", "target": c["name"]}
                else:
                    return {"task": "regression", "target": c["name"]}
        return {"task": "unsupervised", "confidence": 0.5}

    if meta["type"] == "text":
        return {"task": "text_classification", "confidence": 0.8}

    return {"task": "unknown", "confidence": 0.0}
