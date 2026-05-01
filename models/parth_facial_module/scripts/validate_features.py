import pandas as pd
import os

def validate_features(df):
    """
    Validates expected realistic ranges of facial behavioral features.
    
    Expected ranges:
    blink_rate: 5 - 50
    eye_contact_percentage: 0 - 100
    head_movement_frequency: 0 - 60
    emotion_stability: 0 - 1
    emotion_confidence: 0 - 1
    """
    
    ranges = {
        'blink_rate': (5, 50),
        'eye_contact_percentage': (0, 100),
        'head_movement_frequency': (0, 60),
        'emotion_stability': (0.0, 1.0),
        'emotion_confidence': (0.0, 1.0)
    }
    
    is_valid = True
    warnings_count = 0
    
    for idx, row in df.iterrows():
        for feature, (min_val, max_val) in ranges.items():
            if feature in row:
                value = row[feature]
                # Check for out of bounds values (excluding NaNs)
                if pd.notnull(value) and (value < min_val or value > max_val):
                    print(f"Warning: Row {idx} | Feature: '{feature}' | Invalid Value: {value}")
                    is_valid = False
                    warnings_count += 1
                    
    return is_valid, warnings_count

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, 'dataset', 'sample_features.csv')
    
    print(f"Validating dataset from: {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return
        
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded. Total rows: {len(df)}\n")
    
    # Run validation
    is_valid, warnings_count = validate_features(df)
    
    print("-" * 40)
    if is_valid:
        print("Success: Dataset passed validation. All features are within realistic ranges.")
    else:
        print(f"Failed: Dataset validation failed with {warnings_count} out-of-bounds warning(s).")
        print("\nPlease review your dataset construction before analysis.")

if __name__ == "__main__":
    main()
