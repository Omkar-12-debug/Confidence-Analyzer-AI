import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

# Fix pathing
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
HEET_MODULE_DIR = os.path.dirname(CURRENT_DIR)

DATASET_PATH = os.path.join(HEET_MODULE_DIR, "dataset", "audio_dataset.csv")
MODEL_PATH = os.path.join(HEET_MODULE_DIR, "models", "voice_model.pkl")

def train_models(X_train, y_train):
    """Trains Logistic Regression and Random Forest models."""
    print("Training models...")
    
    # 1. Logistic Regression
    lr = LogisticRegression(max_iter=2000, random_state=42)
    lr.fit(X_train, y_train)
    
    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
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
    
    # Save Confusion Matrix Plot
    try:
        class_names = sorted(y_test.unique())
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        
        output_dir = os.path.dirname(MODEL_PATH)
        cm_path = os.path.join(output_dir, f"cm_{model_name.lower().replace(' ', '_')}.png")
        plt.savefig(cm_path)
        print(f"Confusion matrix saved to: {cm_path}")
        plt.close()
    except Exception as e:
        print(f"Note: Could not save confusion matrix plot: {e}")
    
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
    """Selects the best model based on accuracy, with F1 as secondary tie-breaker."""
    print("\n--- Model Comparison Summary ---")
    for res in results:
        print(f"{res['name']}: Accuracy = {res['accuracy']:.4f}, F1-score = {res['f1']:.4f}")
        
    best_res = results[0]
    for res in results[1:]:
        if res['accuracy'] > best_res['accuracy']:
            best_res = res
        elif res['accuracy'] == best_res['accuracy']:
            if res['f1'] > best_res['f1']:
                best_res = res
                
    print(f"\nSelected Model: {best_res['name']}")
    return best_res

def train_model():
    if not os.path.exists(DATASET_PATH):
        raise ValueError(f"Dataset not found at {DATASET_PATH}")
        
    df = pd.read_csv(DATASET_PATH)

    print("\nDataset Loaded:")
    print(df.head())

    if df.empty:
        raise ValueError("Dataset is empty.")

    # Features
    X = df.drop("label", axis=1)

    # Labels
    y = df["label"]

    # Check for only 1 class in y
    unique_classes = y.unique()
    strat = y if len(unique_classes) > 1 else None

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strat
    )

    # Train Models
    lr_model, rf_model = train_models(X_train, y_train)

    results = []
    results.append(evaluate_model(lr_model, X_test, y_test, "Logistic Regression"))
    results.append(evaluate_model(rf_model, X_test, y_test, "Random Forest"))

    best_result = select_best_model(results)

    # Save ONLY best model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best_result['model'], MODEL_PATH)

    print(f"\nBest model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    train_model()