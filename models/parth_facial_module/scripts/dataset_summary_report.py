import pandas as pd
import os

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, 'dataset', 'sample_features.csv')
    analysis_dir = os.path.join(base_dir, 'analysis')
    
    # Ensure analysis directory exists
    os.makedirs(analysis_dir, exist_ok=True)
    
    print(f"Loading dataset from: {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return
        
    df = pd.read_csv(dataset_path)
    
    # Compute summary statistics using pandas
    # specifically: mean, standard deviation, min, max
    summary = df.describe().loc[['mean', 'std', 'min', 'max']]
    
    # Print the summary to the console
    print("\n--- Dataset Summary Statistics ---")
    print(summary.round(3))
    
    # Save the summary as a CSV file
    summary_path = os.path.join(analysis_dir, 'dataset_summary.csv')
    summary.to_csv(summary_path)
    print(f"\nSaved dataset summary to: {summary_path}")
    
    # Generate a correlation matrix and save it
    corr_matrix = df.corr()
    corr_path = os.path.join(analysis_dir, 'feature_correlation_matrix.csv')
    corr_matrix.to_csv(corr_path)
    print(f"Saved feature correlation matrix to: {corr_path}")

if __name__ == "__main__":
    main()
