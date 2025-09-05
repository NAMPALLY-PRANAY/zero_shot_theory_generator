def suggest_pipeline(task, meta):
    """Enhanced pipeline suggestions for various ML tasks."""
    
    # Classification tasks
    if task["task"] == "classification" and meta["type"] == "tabular":
        # Analyze class distribution if available
        target = task.get("target")
        target_col = next((c for c in meta.get("columns", []) if c["name"] == target), None)
        
        # Determine appropriate model based on data characteristics
        n_features = len(meta.get("columns", []))
        n_rows = meta.get("n_rows", 0)
        high_cardinality = len(meta.get("high_cardinality_columns", [])) > 0
        
        if high_cardinality and n_rows > 1000:
            model = "CatBoostClassifier"  # Good for categorical features
        elif n_features > 20 and n_rows > 1000:
            model = "XGBoostClassifier"  # Good for large datasets
        elif n_rows < 100:
            model = "LogisticRegression"  # Simple model for small datasets
        else:
            model = "RandomForestClassifier"  # Good default
            
        return {
            "preprocessing": ["ImputeMissing", "StandardScaler", "OneHotEncoder"],
            "model": model,
            "loss": "CrossEntropy",
            "metrics": ["accuracy", "f1"]
        }
        
    # Binary classification specific
    if task["task"] == "binary_classification":
        return {
            "preprocessing": ["ImputeMissing", "StandardScaler", "OneHotEncoder"],
            "model": "LogisticRegression",
            "loss": "BinaryCrossentropy",
            "metrics": ["accuracy", "precision", "recall", "auc"]
        }
        
    # Regression tasks
    if task["task"] == "regression":
        # Enhanced regression pipeline
        preprocessing = ["ImputeMissing", "StandardScaler"]
        
        # Add one-hot encoding if categorical features present
        categorical_cols = meta.get("categorical_columns", [])
        if categorical_cols:
            preprocessing.append("OneHotEncoder")
            
        # Select model based on data characteristics
        n_features = len(meta.get("columns", []))
        n_rows = meta.get("n_rows", 0)
        
        if n_features > 20 and n_rows > 1000:
            model = "XGBoostRegressor"
        elif n_features > 10:
            model = "RandomForestRegressor" 
        else:
            model = "LinearRegression"
            
        return {
            "preprocessing": preprocessing,
            "model": model,
            "loss": "MSE",
            "metrics": ["rmse", "mae", "r2"]
        }
    
    # Time series forecasting
    if task["task"] == "time_series_forecasting":
        # Time Series specific pipeline
        time_col = task.get("time_col", "Period")
        target = task.get("target", "Revenue")
        
        # Check if we have enough data points for deep learning models
        n_rows = meta.get("n_rows", 0)
        
        if n_rows > 1000:
            return {
                "preprocessing": ["TimeSeriesSplit", "Normalization"],
                "feature_engineering": ["LagFeatures", "RollingStatistics", "SeasonalDecomposition"],
                "model": "LSTM",
                "loss": "MSE",
                "metrics": ["rmse", "mae", "mape"],
                "time_column": time_col,
                "target": target,
                "forecast_horizon": "Auto-detected"
            }
        elif n_rows > 100:
            return {
                "preprocessing": ["TimeSeriesSplit", "Normalization"],
                "feature_engineering": ["LagFeatures", "SeasonalDecomposition"],
                "model": "Prophet",
                "loss": "MSE",
                "metrics": ["rmse", "mae", "mape"],
                "time_column": time_col,
                "target": target,
                "forecast_horizon": "Auto-detected"
            }
        else:
            return {
                "preprocessing": ["TimeSeriesSplit", "Normalization"],
                "feature_engineering": ["LagFeatures"],
                "model": "ARIMA",
                "loss": "MSE",
                "metrics": ["rmse", "mae", "mape"],
                "time_column": time_col,
                "target": target,
                "forecast_horizon": "Auto-detected"
            }
    
    # Anomaly detection
    if task["task"] == "anomaly_detection":
        return {
            "preprocessing": ["StandardScaler"],
            "model": "IsolationForest",
            "metrics": ["precision", "recall", "f1"],
            "alternative_models": ["OneClassSVM", "LocalOutlierFactor"],
            "target": task.get("target", "anomaly")
        }
    
    # Recommendation systems
    if task["task"] == "recommendation":
        if "rating_col" in task:
            return {
                "preprocessing": ["SplitTrainTest"],
                "model": "CollaborativeFiltering",
                "metrics": ["rmse", "mae", "ndcg"],
                "user_col": task.get("user_col"),
                "item_col": task.get("item_col"),
                "rating_col": task.get("rating_col")
            }
        else:
            return {
                "preprocessing": ["SplitTrainTest"],
                "model": "MatrixFactorization",
                "metrics": ["recall@k", "precision@k"],
                "user_col": task.get("user_col"),
                "item_col": task.get("item_col")
            }
    
    # Clustering (specific type of unsupervised)
    if task["task"] == "clustering":
        n_cols = len(meta.get("columns", []))
        return {
            "preprocessing": ["StandardScaler", "PCA"],
            "model": "KMeans" if n_cols < 10 else "DBSCAN",
            "metrics": ["silhouette_score", "davies_bouldin_index"],
            "n_clusters": "auto"
        }
    
    # Image classification
    if task["task"] == "image_classification":
        n_classes = meta.get("n_classes", 2)
        if n_classes > 100:
            model = "EfficientNetB7"  # More complex model for many classes
        elif n_classes > 10:
            model = "ResNet50"
        else:
            model = "ResNet18"  # Simpler model for fewer classes
            
        return {
            "preprocessing": ["Resize(224x224)", "Normalize(ImageNet)", "DataAugmentation"],
            "model": model,
            "loss": "CrossEntropy",
            "metrics": ["accuracy", "top5_accuracy"]
        }
    
    # Object detection
    if task["task"] == "object_detection":
        return {
            "preprocessing": ["Resize(640x640)", "Normalize"],
            "model": "YOLO",
            "metrics": ["mAP", "IoU"],
            "alternative_models": ["FasterRCNN", "SSD"]
        }
    
    # Text classification
    if task["task"] == "text_classification":
        return {
            "preprocessing": ["Tokenize", "Truncate"],
            "model": "DistilBERT",
            "loss": "CrossEntropy",
            "metrics": ["f1", "precision", "recall"]
        }
    
    # Question answering
    if task["task"] == "question_answering":
        return {
            "preprocessing": ["Tokenize", "Truncate"],
            "model": "BERT",
            "metrics": ["exact_match", "f1"],
            "alternative_models": ["RoBERTa", "T5"]
        }
    
    # Named entity recognition
    if task["task"] == "named_entity_recognition":
        return {
            "preprocessing": ["Tokenize", "Truncate"],
            "model": "BERT-NER",
            "metrics": ["token_f1", "span_f1"],
            "alternative_models": ["SpaCy", "Flair"]
        }
    
    # Translation
    if task["task"] == "translation":
        return {
            "preprocessing": ["Tokenize", "Truncate"],
            "model": "MarianMT",
            "metrics": ["bleu", "meteor"],
            "alternative_models": ["T5", "M2M100"]
        }
    
    # Generic unsupervised
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
