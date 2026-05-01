import os
import csv
import json
import sys
import subprocess

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from parth_facial_module.scripts.predict_facial_confidence import predict_confidence

HEET_MODULE_DIR = os.path.join(BASE_DIR, "heet_audio_module")
CSV_PATH = os.path.join(HEET_MODULE_DIR, "dataset", "audio_dataset.csv")
JSON_DIR = os.path.join(BASE_DIR, "bhavesh_visual_module", "data", "processed_features")
VIDEO_DIR = os.path.join(BASE_DIR, "video_data")
VOICE_MODEL_SCRIPT = os.path.join(HEET_MODULE_DIR, "audio", "voice_model.py")

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        return
        
    videos = sorted([f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(".mp4")])
    
    with open(CSV_PATH, "r") as f:
        reader = list(csv.reader(f))
        
    if not reader:
        print("Error: CSV is empty.")
        return
        
    header = reader[0]
    data_rows = reader[1:]
    
    if len(data_rows) != len(videos):
        print("Warning: CSV rows and video count mismatch")
        
    updated_data = []
    labeled_count = 0
    
    for i, row in enumerate(data_rows):
        if i >= len(videos):
            break
            
        video_filename = videos[i]
        json_filename = video_filename.replace(".mp4", ".json")
        json_path = os.path.join(JSON_DIR, json_filename)
        
        if not os.path.exists(json_path):
            print(f"Missing JSON: {json_filename}")
            row[-1] = "neutral"
            updated_data.append(row)
            # Not labeled via JSON, but kept in dataset
            continue
            
        print(f"Matched: {video_filename}")
        
        label = "neutral" # Default fallback
        try:
            with open(json_path, "r") as jf:
                facial_data = json.load(jf)
                
            if isinstance(facial_data, list):
                if len(facial_data) > 0:
                    facial_features = facial_data[-1]
                else: 
                    facial_features = {}
            else:
                facial_features = facial_data
                
            result = predict_confidence(facial_features)
            predicted_class = result.get("confidence_class", "Medium")
            mapping = {"High": "confident", "Medium": "neutral", "Low": "nervous"}
            label = mapping.get(predicted_class, "neutral")
        except Exception as e:
            # Prediction fails -> set label = "neutral"
            label = "neutral"
            
        print(f"Label assigned: {label}")
        
        # Overwrite the empty label column
        row[-1] = label
        updated_data.append(row)
        labeled_count += 1
        
    # Overwrite CSV with updated labels (all rows preserved, missing JSON defaulted to neutral)
    with open(CSV_PATH, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(updated_data)
        
    print(f"\nTotal rows: {len(updated_data)}")
    print(f"Labeled via JSON: {labeled_count}")
    print(f"Fallback (neutral): {len(updated_data) - labeled_count}")
    
    # Validation step
    if labeled_count > 0:
        counts = {"confident": 0, "neutral": 0, "nervous": 0}
        for row in updated_data:
            counts[row[-1]] = counts.get(row[-1], 0) + 1
            
        print("\nClass distribution:")
        for k, v in counts.items():
            print(f"{k}: {v}")
            
        print("\nTraining audio model...")
        # voice_model.py expects specific relative paths, so run from HEET_MODULE_DIR
        try:
            subprocess.run([sys.executable, VOICE_MODEL_SCRIPT], cwd=HEET_MODULE_DIR, check=True)
        except Exception as e:
            print(f"Error executing model training: {e}")
    else:
        print("\nNo labeled data available. Skipping model training to prevent crash.")

if __name__ == "__main__":
    main()
