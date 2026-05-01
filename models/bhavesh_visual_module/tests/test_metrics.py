"""
Test script for behavior metrics module
Bhavesh - Visual Processing Lead
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import time
from src.face_detection import FaceDetector
from src.landmark_extraction import LandmarkExtractor
from src.emotion_recognition import EmotionRecognizer
from src.behavior_metrics import BehaviorMetrics

def test_behavior_metrics():
    print("Testing Behavior Metrics Module...")
    
    # Initialize modules
    face_detector = FaceDetector(method='mediapipe')
    landmark_extractor = LandmarkExtractor()
    emotion_recognizer = EmotionRecognizer()
    metrics = BehaviorMetrics()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Press 'q' to quit")
    print("Collecting metrics...")
    
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect face
        boxes, _ = face_detector.detect_faces(frame)
        
        if boxes:
            # Extract landmarks
            landmarks = landmark_extractor.extract_landmarks(frame)
            
            # Get emotion
            x, y, w, h = boxes[0]
            face_roi = frame[y:y+h, x:x+w]
            
            if face_roi.size > 0:
                emotion, scores = emotion_recognizer.predict_emotion(face_roi)
                confidence = float(np.max(scores))
                
                # Update metrics
                metrics.update_metrics(landmarks, emotion, confidence)
                
                # Draw landmarks
                frame = landmark_extractor.draw_landmarks(frame, landmarks)
        
        # Get and display metrics
        current_metrics = metrics.get_all_metrics()
        
        y_offset = 30
        for key, value in current_metrics.items():
            if isinstance(value, float):
                text = f"{key}: {value:.2f}"
            else:
                text = f"{key}: {value}"
            
            cv2.putText(frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
        
        # Show elapsed time
        elapsed = time.time() - start_time
        cv2.putText(frame, f"Time: {elapsed:.1f}s", (10, y_offset + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow('Behavior Metrics Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final metrics
    print("\nFinal Metrics:")
    final_metrics = metrics.get_all_metrics()
    for key, value in final_metrics.items():
        print(f"{key}: {value}")
    
    print("Test completed")

if __name__ == "__main__":
    test_behavior_metrics()