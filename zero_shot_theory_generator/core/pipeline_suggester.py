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
    return {"pipeline": "unknown"}
