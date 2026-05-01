import os
import sys
import subprocess
import time

# Ensure the scripts directory is in the path for config import
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

import config

def run_labeling():
    """Step 1: Execute label_features.py."""
    print("\n--- Step 1: Labeling ---")
    script_path = os.path.join(config.BASE_DIR, "scripts", "label_features.py")
    try:
        # Run script from its directory to ensure relative imports work
        result = subprocess.run([sys.executable, script_path], 
                                cwd=os.path.join(config.BASE_DIR, "scripts"),
                                capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Labeling failed with error:\n{result.stderr}")
            return False
        print(result.stdout)
        print("Step 1: Labeling completed")
        return True
    except Exception as e:
        print(f"Labeling failed: {e}")
        return False

def run_training():
    """Step 2: Execute train_facial_model.py."""
    print("\n--- Step 2: Training ---")
    script_path = os.path.join(config.BASE_DIR, "scripts", "train_facial_model.py")
    try:
        result = subprocess.run([sys.executable, script_path], 
                                cwd=os.path.join(config.BASE_DIR, "scripts"),
                                capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Training failed with error:\n{result.stderr}")
            return False
        print(result.stdout)
        print("Step 2: Training completed")
        return True
    except Exception as e:
        print(f"Training failed: {e}")
        return False

def validate_outputs():
    """Step 3: Validate generated artifacts."""
    print("\n--- Step 3: Validation ---")
    required_files = [
        ("Labeled Dataset", config.DATASET_LABELED),
        ("Trained Model", config.MODEL_PATH),
        ("Scaler", config.SCALER_PATH)
    ]
    
    all_ok = True
    for name, path in required_files:
        if not os.path.exists(path):
            print(f"Validation failed: {name} not found at {path}")
            all_ok = False
        else:
            print(f"✓ {name} found")
            
    if all_ok:
        print("Step 3: Validation passed")
    return all_ok

def run_test_prediction():
    """Step 4: Run a test prediction using predict_facial_confidence.py."""
    print("\n--- Step 4: Test Prediction ---")
    try:
        # Import prediction function directly for verification
        scripts_path = os.path.join(config.BASE_DIR, "scripts")
        if scripts_path not in sys.path:
            sys.path.append(scripts_path)
            
        from predict_facial_confidence import predict_confidence
        
        # Create sample input using safe copy of defaults
        sample_input = dict(config.DEFAULT_VALUES)
        print(f"Running test prediction with sample input: {sample_input}")
        
        result = predict_confidence(sample_input)
        print(f"Test prediction successful: {result}")
        return True
    except Exception as e:
        print(f"Test prediction failed: {e}")
        return False

def run_full_pipeline():
    """Orchestrates the entire facial module pipeline."""
    start_time = time.time()
    print("==========================================")
    print("STARTING FACIAL MODULE FULL PIPELINE")
    print("==========================================")
    
    # Execute Steps
    success = True
    if not run_labeling():
        print("Pipeline stopped: Labeling failed")
        success = False
        
    if success and not run_training():
        print("Pipeline stopped: Training failed")
        success = False
        
    if success and not validate_outputs():
        print("Pipeline stopped: Validation failed")
        success = False
        
    if success and not run_test_prediction():
        print("Pipeline stopped: Test prediction failed")
        success = False
        
    duration = time.time() - start_time
    print(f"\nPipeline completed in {duration:.2f}s")
    
    if success:
        print("==========================================")
        print("FACIAL MODULE READY FOR INTEGRATION")
        print("==========================================")
    return success

if __name__ == "__main__":
    if not run_full_pipeline():
        sys.exit(1)
