import os, pandas as pd, zipfile, json
import numpy as np
from datetime import datetime

def is_datetime(series):
    """Check if a pandas series contains datetime data."""
    if series.dtype.kind == 'M':  # numpy datetime64
        return True
    
    # Try to convert to datetime
    if series.dtype == object:
        try:
            # Try sample of values for performance
            sample = series.dropna().head(10)
            sample.apply(pd.to_datetime)
            return True
        except:
            pass
    return False

def detect_dataset(path, sample_size=100):
    """Enhanced dataset detection with better feature characterization."""
    if path.endswith(".csv"):
        try:
            df = pd.read_csv(path, nrows=sample_size)
            metadata = analyze_tabular_data(df)
            return metadata
        except pd.errors.ParserError:
            # Try with different delimiters
            try:
                df = pd.read_csv(path, sep='\t', nrows=sample_size)
                metadata = analyze_tabular_data(df)
                return metadata
            except:
                pass

    elif path.endswith((".xlsx", ".xls")):
        try:
            df = pd.read_excel(path, nrows=sample_size)
            metadata = analyze_tabular_data(df)
            return metadata
        except:
            pass

    elif path.endswith(".zip") and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            file_list = z.namelist()
            image_exts = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
            
            # Check if it's an image dataset
            image_files = [f for f in file_list if any(f.lower().endswith(ext) for ext in image_exts)]
            if image_files:
                dirs = [f.split("/")[0] for f in file_list if "/" in f]
                classes = list(set(dirs))
                return {"type": "image_folder", "classes": classes, "n_classes": len(classes)}
            
            # Try to find a CSV in the zip
            csv_files = [f for f in file_list if f.endswith('.csv')]
            if csv_files:
                with z.open(csv_files[0]) as f:
                    df = pd.read_csv(f, nrows=sample_size)
                    metadata = analyze_tabular_data(df)
                    return metadata

    elif path.endswith(".txt"):
        with open(path, "r", encoding="utf-8", errors='ignore') as f:
            lines = f.readlines()[:sample_size]
        
        # Try to parse as JSONL
        try:
            json_objs = [json.loads(line) for line in lines]
            if isinstance(json_objs[0], dict):
                keys = list(json_objs[0].keys())
                return {"type": "jsonl", "keys": keys, "n_lines": len(lines)}
        except Exception:
            pass
        
        # Check if it looks like natural language
        words_per_line = [len(line.split()) for line in lines if line.strip()]
        if words_per_line and np.mean(words_per_line) > 5:
            return {
                "type": "text", 
                "sample": lines[:5], 
                "n_lines": len(lines),
                "avg_words_per_line": np.mean(words_per_line)
            }
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
            metadata = analyze_tabular_data(df)
            return metadata
        except Exception:
            pass
        
        # Try to read as text
        try:
            with open(path, "r", encoding="utf-8", errors='ignore') as f:
                lines = f.readlines()[:sample_size]
            return {"type": "text", "sample": lines[:5], "n_lines": len(lines)}
        except:
            pass
            
        raise ValueError(f"Unsupported dataset format: {path}")

def analyze_tabular_data(df):
    """Perform detailed analysis of tabular data."""
    # Get basic column info
    column_info = []
    
    # Track dataset characteristics
    has_datetime = False
    potential_timeseries = False
    potential_targets = []
    categorical_columns = []
    numeric_columns = []
    high_cardinality_columns = []
    
    for c in df.columns:
        column_data = {
            "name": c,
            "dtype": str(df[c].dtype),
            "n_unique": int(df[c].nunique()),
            "missing": float(df[c].isna().mean())
        }
        
        # Check for datetime columns
        if is_datetime(df[c]):
            has_datetime = True
            column_data["is_datetime"] = True
            
            # Check if sorted - potential time series indicator
            if df[c].dropna().is_monotonic_increasing or df[c].dropna().is_monotonic_decreasing:
                potential_timeseries = True
                column_data["is_sorted"] = True
        
        # Add statistics for numeric columns
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_columns.append(c)
            # Add basic stats
            column_data.update({
                "min": float(df[c].min()) if not pd.isna(df[c].min()) else None,
                "max": float(df[c].max()) if not pd.isna(df[c].max()) else None,
                "mean": float(df[c].mean()) if not pd.isna(df[c].mean()) else None,
                "std": float(df[c].std()) if not pd.isna(df[c].std()) else None
            })
            
            # Check if column looks like a potential target (by name)
            target_indicators = ["target", "class", "label", "y", "price", "sales", "revenue", "cases"]
            if any(indicator in c.lower() for indicator in target_indicators):
                potential_targets.append(c)
                
        # Check for categorical columns
        elif df[c].nunique() < len(df) * 0.5:  # Less than 50% unique values
            categorical_columns.append(c)
            if df[c].nunique() > 10:  # High cardinality categorical
                high_cardinality_columns.append(c)
        
        column_info.append(column_data)
    
    metadata = {
        "type": "tabular",
        "n_rows": len(df),
        "columns": column_info,
        "has_datetime": has_datetime,
        "potential_timeseries": potential_timeseries,
        "potential_targets": potential_targets,
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "high_cardinality_columns": high_cardinality_columns
    }
    
    return metadata
