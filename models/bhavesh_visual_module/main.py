"""
Main entry point for Bhavesh's Visual Processing Module
Bhavesh - Visual Processing Lead
"""

import cv2
import argparse
import time
import json
import os
from datetime import datetime
from src.feature_extractor import FeatureExtractor
from src.video_capture import VideoCapture
from src.utils import create_visualization, export_for_fusion
from config import *

def main():
    parser = argparse.ArgumentParser(description='Visual Processing Module for Confidence Analysis')
    parser.add_argument('--mode', type=str, default='live',
                        choices=['live', 'file', 'test'],
                        help='Mode: live (webcam), file (process video), test (run tests)')
    parser.add_argument('--video', type=str, help='Path to video file (for file mode)')
    parser.add_argument('--duration', type=int, default=RECORDING_DURATION,
                        help='Recording duration in seconds')
    parser.add_argument('--output', type=str, help='Output file for features')
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        run_tests()
        return
    
    # Initialize modules
    video_capture = VideoCapture(source=args.video if args.mode == 'file' else CAMERA_ID)
    feature_extractor = FeatureExtractor()
    
    # Start capture
    video_capture.start_capture()
    
    if args.mode == 'live':
        print(f"[INFO] Starting live capture for {args.duration} seconds")
        print("[INFO] Press 'q' to quit early")
        
        # Start recording
        video_capture.start_recording()
        
        start_time = time.time()
        
        while time.time() - start_time < args.duration:
            frame = video_capture.get_frame()
            if frame is None:
                break
            
            # Process frame
            processed_frame, features = feature_extractor.process_frame(frame)
            
            # Write to recording
            video_capture.write_frame(processed_frame)
            
            # Show frame
            cv2.imshow('Visual Processing Module', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        video_capture.stop_recording()
        
    elif args.mode == 'file':
        print(f"[INFO] Processing video file: {args.video}")
        
        def process_callback(frame, frame_num):
            processed, features = feature_extractor.process_frame(frame)
            if frame_num % 30 == 0:  # Show every 30th frame
                cv2.imshow('Processing', processed)
                cv2.waitKey(1)
            return processed
        
        video_capture.process_video_file(args.video, process_callback, save_frames=True)
    
    # Clean up
    video_capture.release()
    cv2.destroyAllWindows()
    
    # Get summary features
    summary = feature_extractor.get_summary_features()
    
    if summary:
        print("\n=== SUMMARY FEATURES ===")
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
        
        # Save features
        if args.output:
            feature_extractor.save_features(args.output)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            feature_extractor.save_features(os.path.join(FEATURES_DIR, f'summary_{timestamp}.json'))
        
        # Export for fusion
        fusion_file = export_for_fusion(summary, os.path.join(OUTPUTS_DIR, 'fusion_input.json'))
        print(f"\n[INFO] Exported for fusion: {fusion_file}")
        
        # Create visualization
        vis_path = os.path.join(OUTPUTS_DIR, 'visualizations', f'viz_{timestamp}.png')
        create_visualization(feature_extractor.features_history, vis_path)
    
    print("[INFO] Processing completed")

def run_tests():
    """Run all test modules"""
    print("Running all tests...")
    
    # Import test modules
    from tests.test_face_detection import test_face_detection
    from tests.test_emotion import test_emotion_recognition
    from tests.test_metrics import test_behavior_metrics
    
    # Run tests
    test_face_detection()
    test_emotion_recognition()
    test_behavior_metrics()
    
    print("All tests completed")

if __name__ == "__main__":
    main()