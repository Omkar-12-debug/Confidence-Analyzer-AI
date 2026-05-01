import pandas as pd
import numpy as np
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import sys
import os

# Add backend directory to module search path so we can import modules properly
backend_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(backend_path)

import config

def train_model():
    if not config.FUSION_DATASET.exists():
        print(f"Fusion dataset {config.FUSION_DATASET} not found. Run build_fusion_dataset.py first.")
        return
        
    df = pd.read_csv(config.FUSION_DATASET)
    
    # Drop rows with NA in required features
    features = ['voice_confidence_score', 'facial_confidence_score']
    
    # Optional fallback if missing feature
    df = df.dropna(subset=features + ['confidence_label'])
    
    X = df[features]
    y = df['confidence_label']
    
    print(f"Training on {len(df)} samples...")
    print(f"Class distribution:\n{y.value_counts()}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # RandomForest for simple fusion and interpretation
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }
    
    print(f"Metrics: {metrics}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=["Low", "Medium", "High"])
    cm_df = pd.DataFrame(cm, index=["True_Low", "True_Medium", "True_High"], columns=["Pred_Low", "Pred_Medium", "Pred_High"])
    
    # Save artifacts
    joblib.dump(clf, config.FUSION_MODEL_PATH)
    print(f"Model saved to {config.FUSION_MODEL_PATH}")
    
    with open(config.FUSION_METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    cm_df.to_csv(config.FUSION_CONFUSION_MATRIX)
    print(f"Confusion matrix saved to {config.FUSION_CONFUSION_MATRIX}")

if __name__ == "__main__":
    train_model()
