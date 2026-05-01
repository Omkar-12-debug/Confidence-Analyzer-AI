import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    # Set paths relative to this script
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, 'dataset', 'sample_features.csv')
    analysis_dir = os.path.join(base_dir, 'analysis')

    # Ensure analysis directory exists
    os.makedirs(analysis_dir, exist_ok=True)

    # 1. Load the dataset
    print(f"Loading dataset from: {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"Error: Could not find the dataset at {dataset_path}")
        return
        
    df = pd.read_csv(dataset_path)
    
    # Verify expected columns
    expected_cols = [
        'blink_rate', 
        'eye_contact_percentage', 
        'head_movement_frequency', 
        'emotion_stability', 
        'emotion_confidence'
    ]
    
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns in dataset: {missing_cols}")
        return

    # Set modern styling
    sns.set_theme(style="whitegrid")

    print("\nStarting generation of visualizations...")

    # a) Scatter plot: blink_rate vs emotion_confidence
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='blink_rate', y='emotion_confidence', data=df, color='blue', alpha=0.7)
    plt.title('Blink Rate vs Emotion Confidence')
    plt.xlabel('Blink Rate (per minute)')
    plt.ylabel('Emotion Confidence')
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, 'scatter_blink_vs_confidence.png'))
    plt.show() # Display on screen as requested
    
    # b) Scatter plot: eye_contact_percentage vs emotion_confidence
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='eye_contact_percentage', y='emotion_confidence', data=df, color='orange', alpha=0.7)
    plt.title('Eye Contact Percentage vs Emotion Confidence')
    plt.xlabel('Eye Contact Percentage (%)')
    plt.ylabel('Emotion Confidence')
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, 'scatter_eye_contact_vs_confidence.png'))
    plt.show()
    
    # c) Scatter plot: head_movement_frequency vs emotion_confidence
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='head_movement_frequency', y='emotion_confidence', data=df, color='green', alpha=0.7)
    plt.title('Head Movement Frequency vs Emotion Confidence')
    plt.xlabel('Head Movement Frequency')
    plt.ylabel('Emotion Confidence')
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, 'scatter_head_movement_vs_confidence.png'))
    plt.show()

    # d) Pairplot of all features
    print("Generating pairplot...")
    pairplot = sns.pairplot(df[expected_cols], corner=True, diag_kind='kde')
    pairplot.fig.suptitle("Pairplot of Facial Behavioral Features", y=1.02)
    pairplot.savefig(os.path.join(analysis_dir, 'pairplot_all_features.png'))
    plt.show()

    # e) Correlation heatmap between all variables
    print("Generating correlation heatmap...")
    plt.figure(figsize=(10, 8))
    corr_matrix = df[expected_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, square=True)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, 'correlation_heatmap_trends.png'))
    plt.show()

    print(f"\nAll plots saved to the analysis folder at: {analysis_dir}")

if __name__ == "__main__":
    main()
