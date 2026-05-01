"""
Test script for emotion recognition module
Bhavesh - Visual Processing Lead
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from src.face_detection import FaceDetector
from src.emotion_recognition import EmotionRecognizer

def test_emotion_recognition():
    print("Testing Emotion Recognition Module...")
    
    # Initialize modules
    face_detector = FaceDetector(method='mediapipe')
    emotion_recognizer = EmotionRecognizer()
    
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
        
        # Detect face
        boxes, confidences = face_detector.detect_faces(frame)
        
        if boxes:
            # Get largest face
            x, y, w, h = boxes[0]
            face_roi = frame[y:y+h, x:x+w]
            
            if face_roi.size > 0:
                # Predict emotion
                emotion, scores = emotion_recognizer.predict_emotion(face_roi)
                confidence = float(np.max(scores))
                
                # Draw
                frame = emotion_recognizer.draw_emotion(frame, boxes[0], emotion, confidence)
                
                # Show all emotion scores
                y_offset = 60
                for label, score in emotion_recognizer.get_emotion_scores(scores).items():
                    text = f"{label}: {score:.2f}"
                    cv2.putText(frame, text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y_offset += 20
        
        cv2.imshow('Emotion Recognition Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Test completed")

if __name__ == "__main__":
    test_emotion_recognition()