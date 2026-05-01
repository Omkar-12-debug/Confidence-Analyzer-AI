import pandas as pd
import numpy as np
import os

def preprocess_dataset(df):
    """
    Applies basic preprocessing to the feature dataset:
    1. Handles missing values by replacing them with column means.
    2. Normalizes all numeric features between 0 and 1 using min-max scaling.
    """
    
    expected_cols = [
        'blink_rate', 
        'eye_contact_percentage', 
        'head_movement_frequency', 
        'emotion_stability', 
        'emotion_confidence'
    ]
    
    # Check if necessary columns exist
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: The following expected columns are missing: {missing_cols}")
        
    cols_to_process = [col for col in expected_cols if col in df.columns]
    
    processed_df = df.copy()
    
    # 1. Handle missing values
    for col in cols_to_process:
        if processed_df[col].isnull().any():
            mean_val = processed_df[col].mean()
            processed_df[col].fillna(mean_val, inplace=True)
            
    # 2. Min-Max Scaling (Normalization strictly between 0 and 1)
    for col in cols_to_process:
        min_val = processed_df[col].min()
        max_val = processed_df[col].max()
        
        # Avoid division by zero if column has constant value
        if max_val > min_val:
            processed_df[col] = (processed_df[col] - min_val) / (max_val - min_val)
        else:
            processed_df[col] = 0.0 # Or keep it as is, but 0.0 is safe for constant features
            
    return processed_df

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, 'dataset', 'sample_features.csv')
    output_path = os.path.join(base_dir, 'dataset', 'processed_features.csv')
    
    print(f"Loading dataset from: {input_path}")
    if not os.path.exists(input_path):
        print(f"Error: Dataset not found at {input_path}")
        return
        
    df = pd.read_csv(input_path)
    
    print("\n--- Dataset Info Before Preprocessing ---")
    print(f"Shape: {df.shape}")
    print("Descriptive Statistics:")
    print(df.describe().round(3))
    
    # Process
    print("\nApplying preprocessing (Mean imputation & Min-Max scaling)...")
    processed_df = preprocess_dataset(df)
    
    print("\n--- Dataset Info After Preprocessing ---")
    print(f"Shape: {processed_df.shape}")
    print("Descriptive Statistics:")
    print(processed_df.describe().round(3))
    
    # Save
    print(f"\nSaving processed dataset to: {output_path}")
    processed_df.to_csv(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
