import os
import json
import pandas as pd
import config

def safe_float(value, default=0.0):
    """Safely converts a value to float, or returns default if invalid."""
    try:
        if value is None:
            return float(default)
        return float(value)
    except (ValueError, TypeError):
        return float(default)

def load_json_files(directory):
    """Reads all .json files from the specified directory."""
    json_data = []
    if not os.path.exists(directory):
        print(f"Error: Directory {directory} does not exist.")
        return json_data

    # Sort JSON files before processing
    files = sorted([f for f in os.listdir(directory) if f.endswith('.json')])
    print(f"Processing {len(files)} JSON files from: {directory}")

    for file in files:
        file_path = os.path.join(directory, file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Check for list-based summary logs
                if isinstance(data, list) and len(data) > 0:
                    last_entry = data[-1]
                    # Robustness: Ensure last entry is actually a dictionary
                    if isinstance(last_entry, dict):
                        # Track origin of each sample
                        last_entry['source_file'] = file
                        json_data.append(last_entry)
                    else:
                        print(f"Warning: Last entry in {file} is not a dictionary. Skipping.")
                # Check for single-entry dict logs
                elif isinstance(data, dict):
                    data['source_file'] = file
                    json_data.append(data)
                else:
                    print(f"Warning: {file} content is not a list or dict. Skipping.")
                    
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Skipping corrupted file {file}: {e}")
            
    return json_data

def extract_features(data_list):
    """Extracts and cleans features from the list of raw dictionaries."""
    extracted_data = []
    
    for item in data_list:
        features = {}
        # Preserve source tracking
        if 'source_file' in item:
            features['source_file'] = item['source_file']
            
        for col in config.FEATURE_COLUMNS:
            # Fallback to DEFAULT_VALUES if missing
            raw_val = item.get(col, config.DEFAULT_VALUES.get(col))
            
            # Safe float conversion
            val = safe_float(raw_val, default=config.DEFAULT_VALUES.get(col, 0.0))
            
            # Clip value using FEATURE_RANGES
            if col in config.FEATURE_RANGES:
                min_val, max_val = config.FEATURE_RANGES[col]
                features[col] = float(max(min_val, min(max_val, val)))
            else:
                features[col] = val
            
        extracted_data.append(features)
        
    return extracted_data

def assign_label(row):
    """Assigns a label based on the strict rule-based logic in config."""
    # HIGH Logic: Meets ALL High criteria
    is_high = (
        row['eye_contact_percentage'] >= config.THRESHOLDS['HIGH']['eye_contact_min'] and
        row['emotion_stability'] >= config.THRESHOLDS['HIGH']['emotion_stability_min'] and
        row['head_movement_frequency'] <= config.THRESHOLDS['HIGH']['head_movement_max']
    )
    
    if is_high:
        return "High"
        
    # LOW Logic: Meets ANY Low criteria
    is_low = (
        row['eye_contact_percentage'] <= config.THRESHOLDS['LOW']['eye_contact_max'] or
        row['emotion_stability'] <= config.THRESHOLDS['LOW']['emotion_stability_max'] or
        row['head_movement_frequency'] >= config.THRESHOLDS['LOW']['head_movement_min']
    )
    
    if is_low:
        return "Low"
        
    return "Medium"

def validate_dataset(df):
    """Validates data frame integrity and class distribution."""
    if df.empty:
        raise ValueError("CRITICAL: Resulting DataFrame is empty. No valid data samples processed!")

    dist = df[config.LABEL_COLUMN].value_counts().to_dict()
    print("\nClass Distribution:")
    for label in config.REQUIRED_CLASSES:
        count = dist.get(label, 0)
        print(f"  {label}: {count}")
        if count == 0:
            raise ValueError(f"CRITICAL: Missing class '{label}' in generated dataset!")
            
def main():
    # Use centralized BASE_DIR from config
    input_dir = os.path.join(config.BASE_DIR, "..", "bhavesh_visual_module", "data", "processed_features")
    output_path = os.path.join(config.BASE_DIR, "dataset", "processed_features_labeled.csv")
    
    print(f"Starting feature labeling...")

    # 1. Load data
    raw_data = load_json_files(input_dir)
    if not raw_data:
        print("No valid data samples found. Exiting.")
        return

    # 2. Extract and clean
    features_list = extract_features(raw_data)
    
    # 3. Create DataFrame
    df = pd.DataFrame(features_list)
    
    # 4. Labeling and Type Enforcement
    df[config.LABEL_COLUMN] = df.apply(assign_label, axis=1).astype(str)
    
    # 5. Validation
    try:
        validate_dataset(df)
    except ValueError as e:
        print(f"Validation Error: {e}")
        raise
        
    # 6. Save results
    # Reorder columns to include source_file
    cols = ['source_file'] + config.FEATURE_COLUMNS + [config.LABEL_COLUMN]
    df = df.reindex(columns=cols)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\nSuccessfully processed {len(raw_data)} samples.")
    print(f"Labeled dataset saved to: {output_path}")

if __name__ == "__main__":
    main()
