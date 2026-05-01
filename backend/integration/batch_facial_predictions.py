import pandas as pd
import sys
import os

# Add backend directory to module search path so we can import modules properly
backend_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(backend_path)

import config

# Import predict module
parth_module_dir = config.BASE_DIR / "models" / "parth_facial_module" / "scripts"

if 'config' in sys.modules:
    del sys.modules['config']
sys.path.insert(0, str(parth_module_dir))

import predict_facial_confidence

sys.path.pop(0)
sys.modules['config'] = config


def run_batch():
    if not config.RAW_FACIAL_DATASET.exists():
        print(f"Error: dataset {config.RAW_FACIAL_DATASET} not found.")
        return
        
    print(f"Loading facial dataset: {config.RAW_FACIAL_DATASET}")
    df = pd.read_csv(config.RAW_FACIAL_DATASET)
    
    results = []
    
    for idx, row in df.iterrows():
        try:
            input_dict = row.to_dict()
            result = predict_facial_confidence.predict_confidence(input_dict)
            
            results.append({
                "source_file": row.get("source_file", f"row_{idx}"),
                "facial_confidence_score": result.get("facial_confidence_score"),
                "confidence_class": result.get("confidence_class")
            })
            
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            results.append({
                "source_file": row.get("source_file", f"row_{idx}"),
                "facial_confidence_score": None,
                "confidence_class": "error"
            })
            
    out_df = pd.DataFrame(results)
    out_df.to_csv(config.FACIAL_BATCH_CSV, index=False)
    print(f"Batch facial predictions saved to {config.FACIAL_BATCH_CSV}")

if __name__ == "__main__":
    run_batch()
