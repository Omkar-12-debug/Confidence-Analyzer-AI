import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Ensure the scripts directory is in the path for relative imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

import config

def train_models(X_train, y_train):
    """Trains Logistic Regression and Random Forest models."""
    print("Training models...")
    
    # 1. Logistic Regression
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=config.RANDOM_STATE)
    lr.fit(X_train, y_train)
    
    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=config.RANDOM_STATE)
    rf.fit(X_train, y_train)
    
    return lr, rf

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluates a model and prints metrics and confusion matrix."""
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    print(f"\n--- {model_name} Evaluation ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (Macro): {prec:.4f}")
    print(f"Recall (Macro): {rec:.4f}")
    print(f"F1-score (Macro): {f1:.4f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Plot Confusion Matrix (Non-blocking)
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        
        # Save a copy to models directory
        output_dir = os.path.dirname(config.MODEL_PATH)
        os.makedirs(output_dir, exist_ok=True)
        cm_path = os.path.join(output_dir, f"cm_{model_name.lower().replace(' ', '_')}.png")
        plt.savefig(cm_path)
        print(f"Confusion matrix saved to: {cm_path}")
        plt.close()
    except Exception as e:
        print(f"Note: Could not display/save confusion matrix plot: {e}")
    
    return {
        'name': model_name,
        'model': model,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'cm': cm
    }

def select_best_model(results):
    """Selects the best model based on accuracy, with F1 as secondary."""
    print("\n--- Model Comparison Summary ---")
    for res in results:
        print(f"{res['name']}: Accuracy = {res['accuracy']:.4f}, F1-score = {res['f1']:.4f}")
        
    best_res = results[0]
    for res in results[1:]:
        if res['f1'] > best_res['f1']:
            best_res = res
        elif res['f1'] == best_res['f1']:
            # Secondary: Accuracy
            if res['accuracy'] > best_res['accuracy']:
                best_res = res
                
    print(f"\nSelected Model: {best_res['name']}")
    print(f"Final F1-score: {best_res['f1']:.4f}")
    print(f"Final Accuracy: {best_res['accuracy']:.4f}")
    
    if best_res['accuracy'] < config.MIN_ACCURACY_THRESHOLD:
        print(f"WARNING: Model performance is weak (below {config.MIN_ACCURACY_THRESHOLD})")
        
    return best_res

def save_artifacts(model, scaler):
    """Saves the model and scaler to the paths specified in config."""
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(config.SCALER_PATH), exist_ok=True)
    
    joblib.dump(model, config.MODEL_PATH)
    joblib.dump(scaler, config.SCALER_PATH)
    
    print(f"\nBest model saved to: {config.MODEL_PATH}")
    print(f"Scaler saved to: {config.SCALER_PATH}")

def main():
    # 1. Load Data
    data_path = config.DATASET_LABELED
    print(f"Loading labeled dataset from: {data_path}")
    if not os.path.exists(data_path):
        raise ValueError(f"CRITICAL: Labeled dataset not found at {data_path}. Please run label_features.py first.")
        
    df = pd.read_csv(data_path)
    
    # 2. Preprocess (Cleaning & Imputation only)
    print("Preprocessing data (Cleaning)...")
    processed_df = df.copy()
    
    # Imputation using mean (non-inplace assignment)
    for col in config.FEATURE_COLUMNS:
        if col in processed_df.columns:
            if processed_df[col].isnull().any():
                mean_val = processed_df[col].mean()
                processed_df[col] = processed_df[col].fillna(mean_val)
        else:
            # If column missing entirely, fill with default
            processed_df[col] = config.DEFAULT_VALUES.get(col, 0.0)
    
    # 3. Prepare Features (X) and Labels (y)
    X = processed_df[config.FEATURE_COLUMNS]
    
    if config.LABEL_COLUMN not in processed_df.columns:
        raise ValueError(f"CRITICAL: Label column '{config.LABEL_COLUMN}' not found in dataset.")
        
    y = processed_df[config.LABEL_COLUMN].map(config.CLASS_MAPPING)
    
    # Check for NaN in y
    if y.isnull().any():
        print("Warning: Some labels could not be mapped. Dropping unmapped samples.")
        mask = y.notnull()
        X = X[mask]
        y = y[mask]
    
    # Dataset Safety Check
    if X.empty or y.empty:
        raise ValueError("CRITICAL: Dataset is empty after preprocessing/cleaning!")

    # Print Class Distribution before training
    print("\nClass Distribution (Before Training):")
    counts = y.value_counts().sort_index()
    for idx, count in counts.items():
        label_name = config.REVERSE_CLASS_MAPPING.get(int(idx), f"Class_{idx}")
        print(f"  {label_name}: {count}")

    # 4. Train/Test Split with Safe Stratify
    # stratify=y only if multiple classes exist
    unique_classes = np.unique(y)
    strat = y if len(unique_classes) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_STATE, stratify=strat
    )
    
    # 5. Scaling (happens ONLY once here)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Train Models
    lr_model, rf_model = train_models(X_train_scaled, y_train)
    
    # 7. Evaluate Models
    results = []
    results.append(evaluate_model(lr_model, X_test_scaled, y_test, "Logistic Regression"))
    results.append(evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest"))
    
    # 8. Feature Importance for Random Forest
    print("\nFeature Importances (Random Forest):")
    if hasattr(rf_model, 'feature_importances_'):
        importances = rf_model.feature_importances_
        feat_imps = sorted(zip(config.FEATURE_COLUMNS, importances), key=lambda x: x[1], reverse=True)
        for col, imp in feat_imps:
            print(f"  {col}: {imp:.4f}")
    else:
        print("  Note: Selected model does not support feature_importances_.")
        
    # 9. Select Best Model
    best_result = select_best_model(results)
    
    # 10. Save Artifacts
    save_artifacts(best_result['model'], scaler)
    print("\nTraining Pipeline Completed Successfully!")

if __name__ == "__main__":
    main()
