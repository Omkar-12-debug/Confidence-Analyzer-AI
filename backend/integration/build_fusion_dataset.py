import pandas as pd
import sys
import os

# Add backend directory to module search path so we can import modules properly
backend_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(backend_path)

import config

def build_fusion():
    if not config.AUDIO_BATCH_CSV.exists() or not config.FACIAL_BATCH_CSV.exists():
        print("Batch datasets missing. Ensure you ran batch_audio_predictions and batch_facial_predictions.")
        return
        
    print(f"Loading batch audio predictions from {config.AUDIO_BATCH_CSV}")
    audio_df = pd.read_csv(config.AUDIO_BATCH_CSV)
    
    print(f"Loading batch facial predictions from {config.FACIAL_BATCH_CSV}")
    facial_df = pd.read_csv(config.FACIAL_BATCH_CSV)
    
    # Needs ground truth label
    print(f"Loading ground truth from {config.RAW_FACIAL_DATASET}")
    gt_df = pd.read_csv(config.RAW_FACIAL_DATASET)[['source_file', 'confidence_label']]

    # Merge
    fusion_df = pd.merge(audio_df, facial_df, on="source_file", how="inner")
    fusion_df = pd.merge(fusion_df, gt_df, on="source_file", how="inner")

    # Save
    fusion_df.to_csv(config.FUSION_DATASET, index=False)
    print(f"Fusion dataset successful! Saved to {config.FUSION_DATASET}")
    print(f"Total rows: {len(fusion_df)}")

if __name__ == "__main__":
    build_fusion()
