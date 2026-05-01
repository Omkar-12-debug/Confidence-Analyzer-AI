"""
Test script for face detection module
Bhavesh - Visual Processing Lead
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from src.face_detection import FaceDetector

def test_face_detection():
    print("Testing Face Detection Module...")
    
    # Initialize detector
    detector = FaceDetector(method='mediapipe')
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces
        boxes, confidences = detector.detect_faces(frame)
        
        # Draw results
        frame = detector.draw_face_boxes(frame)
        
        # Show info
        if boxes:
            cv2.putText(frame, f"Faces: {len(boxes)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Face Detection Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Test completed")

if __name__ == "__main__":
    test_face_detection()