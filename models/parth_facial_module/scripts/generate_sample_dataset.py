import pandas as pd
import numpy as np
import os

def generate_synthetic_data(num_samples=100):
    """
    Generates a synthetic dataset of facial behavioral features.
    
    Columns:
    - blink_rate (blinks per minute): Typical range 10-25, skewed right (can be up to 40+ when nervous)
    - eye_contact_percentage (0-100): Typical range 30-95
    - head_movement_frequency (movements/min): Typical range 5-30
    - emotion_stability (0.0 - 1.0): Range 0.3 - 0.95
    - emotion_confidence (0.0 - 1.0): Range 0.4 - 0.99
    
    Args:
        num_samples (int): Number of rows to generate.
        
    Returns:
        pd.DataFrame: The generated synthetic dataframe.
    """
    # Seed for reproducibility
    np.random.seed(42)
    
    # Generate realistic ranges using normal/uniform distributions
    # Blink rate: Mean around 18, std dev around 6 (clamped between 5 and 50)
    blink_rate = np.clip(np.random.normal(loc=18, scale=6, size=num_samples), 5, 50).astype(int)
    
    # Eye contact: Mean around 75, std dev around 15 (clamped 0-100)
    eye_contact = np.clip(np.random.normal(loc=75, scale=15, size=num_samples), 0, 100).astype(int)
    
    # Head movement: Mean around 15, std dev around 8 (clamped 0-45)
    head_movement = np.clip(np.random.normal(loc=15, scale=8, size=num_samples), 0, 45).astype(int)
    
    # Emotion stability & confidence: Mostly 0.5 - 0.95
    emotion_stability = np.clip(np.random.normal(loc=0.75, scale=0.15, size=num_samples), 0.0, 1.0)
    emotion_confidence = np.clip(np.random.normal(loc=0.80, scale=0.12, size=num_samples), 0.0, 1.0)
    
    # Create DataFrame
    df = pd.DataFrame({
        'blink_rate': blink_rate,
        'eye_contact_percentage': eye_contact,
        'head_movement_frequency': head_movement,
        'emotion_stability': np.round(emotion_stability, 3),
        'emotion_confidence': np.round(emotion_confidence, 3)
    })
    
    return df

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, 'dataset')
    output_path = os.path.join(dataset_dir, 'sample_features.csv')
    
    # Ensure directory exists
    os.makedirs(dataset_dir, exist_ok=True)
    
    print("Generating synthetic dataset (100 rows)...")
    df = generate_synthetic_data(100)
    
    print(f"Saving dataset to: {output_path}")
    df.to_csv(output_path, index=False)
    
    print("\nSample Preview:")
    print(df.head())
    print(f"\nTotal rows generated: {len(df)}")

if __name__ == "__main__":
    main()
