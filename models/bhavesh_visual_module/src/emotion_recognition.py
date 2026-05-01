"""
Emotion recognition module using pretrained model
Bhavesh - Visual Processing Lead
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import urllib.request
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

class EmotionRecognizer:
    def __init__(self, model_path=EMOTION_MODEL_PATH):
        """Initialize emotion recognition model"""
        self.model_path = model_path
        self.model = None
        self.emotion_labels = EMOTION_LABELS
        self.input_size = (48, 48)
        
        self._load_or_download_model()
        print("[INFO] Emotion recognizer initialized")
    
    def _load_or_download_model(self):
        """Load pretrained model or download if not available"""
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
            print(f"[INFO] Loaded emotion model from {self.model_path}")
        else:
            print("[INFO] Model not found. Downloading...")
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Try multiple sources
            urls = [
                "https://github.com/serengil/tensorflow-101/raw/master/model/fer2013_mini_XCEPTION.102-0.66.hdf5",
                "https://github.com/Shireen1910/Emotion-Detection/raw/master/model.hdf5"
            ]
            
            downloaded = False
            for url in urls:
                try:
                    urllib.request.urlretrieve(url, self.model_path)
                    downloaded = True
                    print(f"[INFO] Downloaded from {url}")
                    break
                except:
                    continue
            
            if downloaded:
                self.model = load_model(self.model_path)
                print("[INFO] Model loaded successfully")
            else:
                print("[WARNING] Could not download model. Will use fallback method.")
                self.model = self._create_fallback_model()
    
    def _create_fallback_model(self):
        """Create a simple CNN as fallback if download fails"""
        print("[INFO] Creating fallback emotion model...")
        from tensorflow.keras import layers, models
        
        model = models.Sequential([
            layers.Input(shape=(48, 48, 1)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(7, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        print("[INFO] Fallback model created")
        return model
    
    def preprocess_face(self, face_roi):
        """Preprocess face region for emotion recognition"""
        if len(face_roi.shape) == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_roi
        
        resized = cv2.resize(gray, self.input_size)
        normalized = resized.astype('float32') / 255.0
        
        input_data = np.expand_dims(normalized, axis=0)
        input_data = np.expand_dims(input_data, axis=-1)
        
        return input_data
    
    def predict_emotion(self, face_roi):
        """Predict emotion from face region"""
        if self.model is None or face_roi.size == 0:
            return 'Neutral', np.zeros(len(self.emotion_labels))
        
        try:
            input_data = self.preprocess_face(face_roi)
            predictions = self.model.predict(input_data, verbose=0)[0]
            
            emotion_idx = np.argmax(predictions)
            emotion_label = self.emotion_labels[emotion_idx]
            confidence = predictions[emotion_idx]
            
            return emotion_label, predictions
        except Exception as e:
            print(f"[ERROR] Emotion prediction failed: {e}")
            return 'Neutral', np.zeros(len(self.emotion_labels))
    
    def get_emotion_scores(self, predictions):
        """Get dictionary of emotion scores"""
        return {label: float(score) for label, score in zip(self.emotion_labels, predictions)}
    
    def draw_emotion(self, frame, face_box, emotion_label, confidence):
        """Draw emotion label on frame"""
        x, y, w, h = face_box
        
        text = f"{emotion_label}: {confidence:.2f}"
        cv2.putText(frame, text, (x, y - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        color_map = {
            'Happy': (0, 255, 0),
            'Sad': (255, 0, 0),
            'Angry': (0, 0, 255),
            'Fear': (128, 0, 128),
            'Disgust': (0, 128, 0),
            'Surprise': (255, 255, 0),
            'Neutral': (128, 128, 128)
        }
        
        color = color_map.get(emotion_label, (255, 255, 255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        return frame