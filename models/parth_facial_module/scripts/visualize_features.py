import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(csv_path):
    """Loads feature dataset from CSV."""
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        print(f"File not found: {csv_path}")
        return None

def main():
    # Set paths relative to this script
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, 'dataset', 'sample_features.csv')
    analysis_dir = os.path.join(base_dir, 'analysis')

    # Ensure analysis directory exists
    os.makedirs(analysis_dir, exist_ok=True)

    # Load data
    print(f"Attempting to load data from {dataset_path}")
    df = load_data(dataset_path)

    if df is not None:
        print("Data loaded successfully. Generating visualizations...")

        # Set modern styling
        sns.set_theme(style="whitegrid")

        # 1. Histogram of blink rate
        plt.figure(figsize=(8, 6))
        sns.histplot(df['blink_rate'], kde=True, bins=20)
        plt.title('Distribution of Blink Rate')
        plt.xlabel('Blink Rate')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'hist_blink_rate.png'))
        plt.close()

        # 2. Histogram of eye contact percentage
        plt.figure(figsize=(8, 6))
        sns.histplot(df['eye_contact_percentage'], kde=True, bins=20, color='orange')
        plt.title('Distribution of Eye Contact Percentage')
        plt.xlabel('Eye Contact Percentage')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'hist_eye_contact.png'))
        plt.close()

        # 3. Boxplot of head movement frequency
        plt.figure(figsize=(8, 6))
        sns.boxplot(y=df['head_movement_frequency'], color='lightgreen')
        plt.title('Boxplot of Head Movement Frequency')
        plt.ylabel('Head Movement Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'boxplot_head_movement.png'))
        plt.close()

        # 4. Correlation heatmap
        expected_features = [
            'blink_rate', 
            'eye_contact_percentage', 
            'head_movement_frequency', 
            'emotion_stability', 
            'emotion_confidence'
        ]
        
        # Filter for columns that actually exist in the DataFrame
        features_to_plot = [f for f in expected_features if f in df.columns]
        
        if len(features_to_plot) >= 2:
            plt.figure(figsize=(10, 8))
            corr_matrix = df[features_to_plot].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
            plt.title('Correlation Heatmap structure')
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, 'correlation_heatmap.png'))
            plt.close()
            
        print(f"Done! Visualizations saved to {analysis_dir}")
    else:
        print("Visualization skipped. Please place your CSV file at dataset/features.csv")

if __name__ == "__main__":
    main()
