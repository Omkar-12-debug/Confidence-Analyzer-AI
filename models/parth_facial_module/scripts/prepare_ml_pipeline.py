import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def prepare_and_train():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, 'dataset', 'processed_features.csv')
    
    # 1. Load dataset
    print(f"Loading processed dataset from: {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please run preprocess_features.py first.")
        return
        
    df = pd.read_csv(dataset_path)
    
    # 2. Prepare features (X) and future label room
    feature_cols = [
        'blink_rate', 
        'eye_contact_percentage', 
        'head_movement_frequency', 
        'emotion_stability', 
        'emotion_confidence'
    ]
    
    # Separate features
    X = df[feature_cols]
    
    print("\nPreparing ML pipeline...")
    # 4. Generate temporary dummy labels for testing
    # Since real labels are not yet available, we create a binary 'confidence_label' (0 or 1)
    np.random.seed(42)  # For reproducibility
    y = np.random.randint(0, 2, size=len(df))
    
    # Leaving room for the future label column in the dataframe
    df['confidence_label'] = y
    
    # 3. Implement placeholder ML pipeline using scikit-learn
    print("Splitting dataset into training and testing sets...")
    
    # train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Print statistics
    print(f"\n--- Pipeline Summary ---")
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    print(f"Features used: {len(feature_cols)} ({', '.join(feature_cols)})")
    
    print("\nTraining RandomForestClassifier pipeline...")
    # RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    print("Evaluating model...")
    # Evaluation with accuracy score
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Accuracy on test set: {accuracy * 100:.2f}%")
    print("Note: This is a prototype pipeline running on generated dummy labels.")

def main():
    prepare_and_train()

if __name__ == "__main__":
    main()
