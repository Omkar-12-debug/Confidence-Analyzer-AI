import os
import sys
import json
import numpy as np

# Adjust sys.path to include current module
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.feature_extractor import FeatureExtractor
from config import FEATURES_DIR

VIDEO_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "video_data"))

def validate_blinks_variance(history):
    """Ensure blink_rate is not all zero across first 5 videos."""
    if len(history) < 5:
        return True
    
    blink_rates = [h.get('blink_rate', 0) for h in history]
    if all(b == 0 for b in blink_rates):
        print("\nCRITICAL VALIDATION ERROR: All blink_rate == 0 across the first 5 videos.")
        print("Blink detection failed — EAR threshold or logic incorrect")
        return False
        
    return True

def main():
    if not os.path.exists(VIDEO_DIR):
        print(f"Error: Video directory {VIDEO_DIR} not found.")
        return

    os.makedirs(FEATURES_DIR, exist_ok=True)

    # STEP X - INPUT PROCESSING FIX: Remove "_part" constraint
    videos = sorted([f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(".mp4")])
    
    if not videos:
        print("No video files found.")
        return

    print(f"Found {len(videos)} clips to extract features from.")

    history = []
    processed_count = 0
    import cv2

    for video_name in videos:
        video_path = os.path.join(VIDEO_DIR, video_name)
        json_name = video_name.replace(".mp4", ".json")
        output_json_path = os.path.join(FEATURES_DIR, json_name)

        # Skip already processed only if we want to save time, 
        # but user goal says "Process ALL video files". Overwriting for fresh state.
        if os.path.exists(output_json_path):
            os.remove(output_json_path)

        print(f"\nProcessing: {video_name}...")
        
        extractor = FeatureExtractor()
        capture = cv2.VideoCapture(video_path)
        
        frames_processed = 0
        frames_with_face = 0
        frame_counter = 0
        
        ears = [] # For debug print

        try:
            while True:
                ret, frame = capture.read()
                if not ret or frame is None:
                    break
                
                frame_counter += 1
                # Still skipping for speed but processing enough for blink logic?
                # Actually blink logic (2 frames) needs high temporal resolution.
                # If we skip every 5 frames, we lose blink sensitivity.
                # Processing ALL frames for better blink detection.
                
                # frame_skip intentionally removed to ensure EAR accuracy
                
                _, features = extractor.process_frame(frame)
                
                # Sample EAR for debug (if landmarks available)
                if 'ear_history' in dir(extractor.behavior_metrics) and len(extractor.behavior_metrics.ear_history) > 0:
                    ears.append(extractor.behavior_metrics.ear_history[-1])

                frames_processed += 1
                if features.get('face_detected', False):
                    frames_with_face += 1
                    
        except Exception as e:
            print(f"Error processing {video_name}: {e}")
        finally:
            capture.release()

        print(f"Frames processed: {frames_processed}")
        print(f"Frames with face: {frames_with_face}")
        
        if frames_with_face == 0:
            print(f"CRITICAL ERROR: No faces detected in {video_name}. Skipping record.")
            continue # Moved from return to continue to try other videos unless user said stop

        summary = extractor.get_summary_features()
        if summary:
            # Save single json explicitly mapping to the video name
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump([summary], f, indent=4)
                
            processed_count += 1
            history.append(summary)
            
            # STEP X - DEBUG (MANDATORY) for first 5 videos
            if processed_count <= 5:
                # Sample some EAR values
                sample_count = 10
                step = max(1, len(ears) // sample_count)
                sample_ears = [f"{e:.4f}" for e in ears[::step][:sample_count]]
                print(f"Video: {video_name}")
                print(f"  EAR values (sampled): {sample_ears}")
                print(f"  blink_count: {extractor.behavior_metrics.blink_count}")
                print(f"  duration: {summary.get('analysis_duration', 0):.2f}s")
                print(f"  final blink_rate: {summary.get('blink_rate', 0):.4f}")
                
            # STEP X - VALIDATION CONDITION after processing 5 videos
            if processed_count == 5:
                if not validate_blinks_variance(history):
                    return # STOP execution as requested

        else:
            print(f"Warning: No summary features generated for {video_name}.")

    print(f"\nBatch processing complete. Total processed: {processed_count}")

if __name__ == "__main__":
    main()
