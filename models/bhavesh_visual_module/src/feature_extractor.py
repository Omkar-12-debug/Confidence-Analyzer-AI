"""
Main feature extractor that combines all modules
Bhavesh - Visual Processing Lead
"""

import cv2
import numpy as np
import json
import os
from datetime import datetime
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from .face_detection import FaceDetector
from .landmark_extraction import LandmarkExtractor
from .emotion_recognition import EmotionRecognizer
from .behavior_metrics import BehaviorMetrics

class FeatureExtractor:
    def __init__(self):
        """Initialize all visual processing modules"""
        print("[INFO] Initializing Feature Extractor...")
        
        self.face_detector = FaceDetector(method='mediapipe')
        self.landmark_extractor = LandmarkExtractor()
        self.emotion_recognizer = EmotionRecognizer()
        self.behavior_metrics = BehaviorMetrics()
        
        self.frame_count = 0
        self.features_history = []
        
        print("[INFO] Feature Extractor initialized successfully")
    
    def process_frame(self, frame):
        """
        Process single frame and extract all features
        Returns: processed frame with visualizations, features dictionary
        """
        self.frame_count += 1
        features = {
            'frame_number': self.frame_count,
            'timestamp': datetime.now().isoformat(),
            'face_detected': False
        }
        
        # Detect faces
        face_boxes, face_confidences = self.face_detector.detect_faces(frame)
        
        if face_boxes:
            # Get largest face (assuming single person)
            largest_face, confidence = self.face_detector.get_largest_face()
            features['face_detected'] = True
            features['face_confidence'] = confidence
            
            # Extract landmarks
            landmarks_dict = self.landmark_extractor.extract_landmarks(frame)
            features['landmarks_detected'] = landmarks_dict['face_detected']
            
            # Extract face ROI for emotion recognition
            x, y, w, h = largest_face
            face_roi = frame[y:y+h, x:x+w]
            
            if face_roi.size > 0:
                # Recognize emotion every 5 frames
                if self.frame_count % 5 == 0 or not hasattr(self, 'last_emotion_label'):
                    try:
                        self.last_emotion_label, self.last_emotion_scores = self.emotion_recognizer.predict_emotion(face_roi)
                        self.last_emotion_scores_dict = self.emotion_recognizer.get_emotion_scores(self.last_emotion_scores)
                    except Exception as e:
                        print(f"Emotion detection failed: {e}")
                        if not hasattr(self, 'last_emotion_label'):
                            self.last_emotion_label = 'Neutral'
                            self.last_emotion_scores = np.zeros(7)
                            self.last_emotion_scores_dict = {}
                
                features['emotion'] = self.last_emotion_label
                features['emotion_scores'] = self.last_emotion_scores_dict
                
                # Update behavior metrics
                self.behavior_metrics.update_metrics(
                    landmarks_dict, 
                    self.last_emotion_label, 
                    float(np.max(self.last_emotion_scores))
                )
                
                # Draw on frame
                frame = self.face_detector.draw_face_boxes(frame)
                frame = self.landmark_extractor.draw_landmarks(frame, landmarks_dict)
                frame = self.emotion_recognizer.draw_emotion(
                    frame, largest_face, self.last_emotion_label, float(np.max(self.last_emotion_scores))
                )
        
        # Get current metrics
        current_metrics = self.behavior_metrics.get_all_metrics()
        features.update(current_metrics)
        
        # Draw metrics on frame
        frame = self._draw_metrics(frame, current_metrics)
        
        # Store features
        self.features_history.append(features)
        
        return frame, features
    
    def _draw_metrics(self, frame, metrics):
        """Draw metrics on frame"""
        y_offset = 30
        for key, value in metrics.items():
            if isinstance(value, float):
                text = f"{key}: {value:.2f}"
            else:
                text = f"{key}: {value}"
            
            cv2.putText(frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
        
        return frame
    
    def save_features(self, filename=None):
        """Save extracted features to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(FEATURES_DIR, f'features_{timestamp}.json')
        
        with open(filename, 'w') as f:
            json.dump(self.features_history, f, indent=2)
        
        print(f"[INFO] Features saved to {filename}")
        return filename
    
    def get_summary_features(self):
        """Get summary features for fusion model"""
        if not self.features_history:
            return None
        
        # Get latest metrics
        latest = self.behavior_metrics.get_all_metrics()
        
        # Format for Omkar's fusion module
        summary = {
            'facial_emotion': latest['facial_emotion'],
            'emotion_confidence': latest['emotion_confidence'],
            'emotion_stability': latest['emotion_stability'],
            'blink_rate': latest['blink_rate'],
            'eye_contact_percentage': latest['eye_contact_percentage'],
            'head_movement_frequency': latest['head_movement_frequency'],
            'face_detected': any(f['face_detected'] for f in self.features_history[-30:]),
            'total_frames_processed': self.frame_count,
            'analysis_duration': latest['analysis_duration']
        }
        
        return summary
    
    def reset(self):
        """Reset all modules"""
        self.frame_count = 0
        self.features_history = []
        self.behavior_metrics.reset()
        print("[INFO] Feature extractor reset")