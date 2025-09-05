def infer_task(meta):
    """Improved task inference with better domain detection."""
    
    # Handle image data
    if meta["type"] == "image_folder":
        # Check if likely object detection (many classes) vs classification
        n_classes = meta.get("n_classes", 0)
        if n_classes > 20:
            return {"task": "object_detection", "confidence": 0.8}
        return {"task": "image_classification", "confidence": 0.95}

    # Handle tabular data
    if meta["type"] == "tabular":
        # Check for image columns (embedded or path-based)
        image_cols = [c for c in meta["columns"] if c["name"].lower() in ("image_path", "image_url", "img", "image")]
        if image_cols:
            return {"task": "image_classification", "input": image_cols[0]["name"], "confidence": 0.9}
        
        # Time series detection with enhanced logic
        if meta.get("potential_timeseries", False) or meta.get("has_datetime", False):
            time_cols = [c for c in meta["columns"] 
                        if c.get("is_datetime", False) or c["name"].lower() in 
                        ("date", "time", "datetime", "timestamp", "year", "month", "day", "period", "week")]
            
            if time_cols:
                # Try to find target columns
                potential_targets = meta.get("potential_targets", [])
                if not potential_targets:
                    # Look by name
                    potential_targets = [c["name"] for c in meta["columns"] 
                                      if c["name"].lower() in ("revenue", "sales", "price", "value", "demand", "volume", 
                                                            "consumption", "production", "stock", "return")]
                
                if potential_targets:
                    return {
                        "task": "time_series_forecasting", 
                        "time_col": time_cols[0]["name"],
                        "target": potential_targets[0], 
                        "confidence": 0.9
                    }
                else:
                    # Use first numeric column that's not the time column as target
                    numeric_cols = meta.get("numeric_columns", [])
                    time_col_names = [c["name"] for c in time_cols]
                    potential_targets = [c for c in numeric_cols if c not in time_col_names]
                    
                    if potential_targets:
                        return {
                            "task": "time_series_forecasting",
                            "time_col": time_cols[0]["name"],
                            "target": potential_targets[0],
                            "confidence": 0.85
                        }
        
        # Anomaly detection - if column named "anomaly", "outlier", "fraud", etc.
        anomaly_cols = [c for c in meta["columns"] 
                       if any(term in c["name"].lower() for term in ["anomaly", "outlier", "fraud", "error"])]
        if anomaly_cols:
            return {
                "task": "anomaly_detection",
                "target": anomaly_cols[0]["name"],
                "confidence": 0.9
            }
        
        # Recommendation - if columns suggest user-item interactions
        user_cols = [c for c in meta["columns"] if c["name"].lower() in ("user", "user_id", "customer", "customer_id")]
        item_cols = [c for c in meta["columns"] if c["name"].lower() in ("item", "item_id", "product", "product_id")]
        rating_cols = [c for c in meta["columns"] if c["name"].lower() in ("rating", "score", "preference", "rank")]
        
        if user_cols and item_cols:
            task_info = {
                "task": "recommendation",
                "user_col": user_cols[0]["name"],
                "item_col": item_cols[0]["name"],
                "confidence": 0.85
            }
            if rating_cols:
                task_info["rating_col"] = rating_cols[0]["name"]
            return task_info
        
        # Common target column names
        regression_target_names = ["target", "y", "value", "price", "sales", "revenue", "cases", "count", "amount"]
        classification_target_names = ["target", "label", "class", "y", "category", "type", "result"]
        
        # First look for explicit target columns
        for c in meta["columns"]:
            col_name = c["name"].lower()
            
            # Check for regression targets
            if col_name in regression_target_names or col_name.endswith("_target"):
                if c["dtype"].startswith(("float", "int")):
                    return {"task": "regression", "target": c["name"], "confidence": 0.9}
                
            # Check for classification targets
            if col_name in classification_target_names or col_name.endswith(("_class", "_label")):
                if c.get("n_unique", 0) <= 50:  # Classification typically has limited classes
                    # Determine if binary or multi-class
                    task_type = "binary_classification" if c.get("n_unique", 0) <= 2 else "classification"
                    return {"task": task_type, "target": c["name"], "confidence": 0.9}
        
        # If no explicit targets found, try to infer from data characteristics
        # Look for numeric columns that might be targets
        numeric_cols = [c for c in meta["columns"] if c["dtype"].startswith(("float", "int"))]
        
        # Check for common regression scenarios (like forecasting, prediction)
        for c in numeric_cols:
            col_name = c["name"].lower()
            # Columns likely to be regression targets
            if any(term in col_name for term in ["cases", "count", "amount", "price", "value", "rate"]):
                return {"task": "regression", "target": c["name"], "confidence": 0.85}
        
        # If still no target found but we have columns named "labels" with only 1 unique value
        # and "cases" with multiple values, assume "cases" is the regression target
        labels_col = next((c for c in meta["columns"] if c["name"].lower() == "labels" and c.get("n_unique", 0) == 1), None)
        cases_col = next((c for c in meta["columns"] if c["name"].lower() == "cases" and c.get("n_unique", 0) > 1), None)
        if labels_col and cases_col:
            return {"task": "regression", "target": "cases", "confidence": 0.9}
        
        # Check for clustering task (no clear target, mostly numeric features)
        if len(numeric_cols) > 3 and len(meta.get("categorical_columns", [])) < len(numeric_cols):
            return {"task": "clustering", "confidence": 0.7}
            
        return {"task": "unsupervised", "confidence": 0.5}

    # Handle text data
    if meta["type"] == "text":
        # Check for different NLP tasks
        sample_text = " ".join(meta.get("sample", []))
        
        # Check for question-answering pattern
        if "?" in sample_text and len(sample_text.split("?")) > 1:
            return {"task": "question_answering", "confidence": 0.7}
            
        # Check for named entity recognition (look for capitalized words)
        words = sample_text.split()
        capitalized = [w for w in words if w[0].isupper()]
        if len(capitalized) / max(1, len(words)) > 0.15:  # At least 15% capitalized words
            return {"task": "named_entity_recognition", "confidence": 0.6}
            
        # Default to text classification
        return {"task": "text_classification", "confidence": 0.8}

    # Handle JSON data
    if meta["type"] in ["json", "json_list", "jsonl"]:
        keys = meta.get("keys", [])
        text_indicators = ["text", "content", "body", "description", "title"]
        label_indicators = ["label", "class", "category", "sentiment"]
        
        # Check for text classification in JSON
        if any(k in keys for k in text_indicators) and any(k in keys for k in label_indicators):
            return {"task": "text_classification", "confidence": 0.8}
            
        # Check for translation
        if any(k.startswith("text_") or k.endswith("_text") for k in keys):
            return {"task": "translation", "confidence": 0.7}
            
    return {"task": "unknown", "confidence": 0.0}
