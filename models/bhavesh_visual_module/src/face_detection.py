"""
Face detection module using MediaPipe and OpenCV
Bhavesh - Visual Processing Lead
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


def _resolve_haarcascade_path(xml_filename):
    """
    Resolve the full path to a Haar cascade XML file.

    Search order:
      1. cv2.data.haarcascades  (normal pip-installed OpenCV)
      2. sys._MEIPASS/cv2/data/ (PyInstaller frozen bundle)
      3. Alongside cv2.__file__  (some wheel layouts)
    """
    candidates = []

    if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
        candidates.append(os.path.join(cv2.data.haarcascades, xml_filename))

    if getattr(sys, 'frozen', False):
        meipass = getattr(sys, '_MEIPASS', '')
        candidates.append(os.path.join(meipass, 'cv2', 'data', xml_filename))

    cv2_dir = os.path.dirname(os.path.abspath(cv2.__file__))
    candidates.append(os.path.join(cv2_dir, 'data', xml_filename))

    for path in candidates:
        if os.path.isfile(path):
            return path

    raise FileNotFoundError(
        f"Haar cascade '{xml_filename}' not found. Searched: {candidates}"
    )


class FaceDetector:
    def __init__(self, method='mediapipe'):
        """
        Initialize face detector
        method: 'mediapipe' or 'haar'
        """
        self.method = method
        self.face_detected = False
        self.face_boxes = []
        self.face_confidence = []
        
        if method == 'mediapipe':
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=FACE_DETECTION_CONFIDENCE
            )
            self.mp_drawing = mp.solutions.drawing_utils
            
        elif method == 'haar':
            cascade_path = _resolve_haarcascade_path('haarcascade_frontalface_default.xml')
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                raise RuntimeError(
                    f"CascadeClassifier loaded but is empty for: {cascade_path}"
                )
        
        print(f"[INFO] Face detector initialized using {method}")
    
    def detect_faces_mediapipe(self, frame):
        """Detect faces using MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        self.face_boxes = []
        self.face_confidence = []
        
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                width = int(bboxC.width * w)
                height = int(bboxC.height * h)
                
                self.face_boxes.append([x, y, width, height])
                self.face_confidence.append(detection.score[0])
        
        self.face_detected = len(self.face_boxes) > 0
        return self.face_boxes, self.face_confidence
    
    def detect_faces_haar(self, frame):
        """Detect faces using Haar cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        self.face_boxes = faces.tolist() if len(faces) > 0 else []
        self.face_confidence = [1.0] * len(faces)
        
        self.face_detected = len(self.face_boxes) > 0
        return self.face_boxes, self.face_confidence
    
    def detect_faces(self, frame):
        """Main face detection method"""
        if self.method == 'mediapipe':
            return self.detect_faces_mediapipe(frame)
        else:
            return self.detect_faces_haar(frame)
    
    def get_largest_face(self):
        """Get the largest face detected"""
        if not self.face_boxes:
            return None, None
        
        areas = [w * h for (x, y, w, h) in self.face_boxes]
        largest_idx = np.argmax(areas)
        
        return self.face_boxes[largest_idx], self.face_confidence[largest_idx]
    
    def draw_face_boxes(self, frame, color=(0, 255, 0), thickness=2):
        """Draw bounding boxes around detected faces"""
        for i, (x, y, w, h) in enumerate(self.face_boxes):
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            
            if i < len(self.face_confidence):
                conf_text = f"{self.face_confidence[i]:.2f}"
                cv2.putText(frame, conf_text, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame