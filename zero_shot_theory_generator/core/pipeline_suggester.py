def suggest_pipeline(task, meta):
    if task["task"] == "classification" and meta["type"] == "tabular":
        return {
            "preprocessing": ["ImputeMissing", "StandardScaler"],
            "model": "RandomForestClassifier",
            "loss": "CrossEntropy",
            "metrics": ["accuracy", "f1"]
        }
    if task["task"] == "regression":
        return {
            "preprocessing": ["ImputeMissing", "StandardScaler"],
            "model": "LinearRegression",
            "loss": "MSE",
            "metrics": ["rmse", "r2"]
        }
    if task["task"] == "image_classification":
        return {
            "preprocessing": ["Resize(224x224)", "Normalize(ImageNet)"],
            "model": "ResNet18",
            "loss": "CrossEntropy",
            "metrics": ["accuracy"]
        }
    if task["task"] == "text_classification":
        return {
            "preprocessing": ["Tokenize", "Truncate"],
            "model": "DistilBERT",
            "loss": "CrossEntropy",
            "metrics": ["f1", "precision", "recall"]
        }
    if task["task"] == "unsupervised":
        n_cols = len(meta.get("columns", []))
        pipeline = {
            "preprocessing": ["StandardScaler"] if n_cols > 0 else [],
            "clustering": "KMeans" if n_cols < 10 else "DBSCAN",
            "dimensionality_reduction": "PCA" if n_cols < 20 else "t-SNE",
            "metrics": ["silhouette_score", "calinski_harabasz"]
        }
        return pipeline
    return {"pipeline": "unknown"}
