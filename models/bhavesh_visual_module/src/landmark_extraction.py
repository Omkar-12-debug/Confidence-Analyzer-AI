"""
Facial landmark extraction module using MediaPipe
Bhavesh - Visual Processing Lead
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class LandmarkExtractor:
    def __init__(self):
        """Initialize MediaPipe Face Mesh"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Define landmark indices for EAR calculation (P1, P2, P3, P4, P5, P6 order)
        self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        self.LEFT_IRIS = [468, 469, 470, 471]
        self.RIGHT_IRIS = [472, 473, 474, 475]
        self.MOUTH_INDICES = [61, 291, 13, 14]
        self.NOSE_TIP = 1
        self.FOREHEAD = 10
        self.CHIN = 152
        
        print("[INFO] Landmark extractor initialized")
    
    def extract_landmarks(self, frame):
        """Extract facial landmarks from frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        landmarks_dict = {
            'all_landmarks': None,
            'left_eye': [],
            'right_eye': [],
            'left_iris': [],
            'right_iris': [],
            'mouth': [],
            'nose_tip': None,
            'forehead': None,
            'chin': None,
            'face_detected': False
        }
        
        if results.multi_face_landmarks:
            landmarks_dict['face_detected'] = True
            face_landmarks = results.multi_face_landmarks[0]
            
            h, w = frame.shape[:2]
            all_points = []
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                z = landmark.z
                all_points.append([x, y, z])
            
            landmarks_dict['all_landmarks'] = np.array(all_points)
            
            # Extract specific features
            landmarks_dict['left_eye'] = [all_points[i] for i in self.LEFT_EYE_INDICES]
            landmarks_dict['right_eye'] = [all_points[i] for i in self.RIGHT_EYE_INDICES]
            landmarks_dict['left_iris'] = [all_points[i] for i in self.LEFT_IRIS]
            landmarks_dict['right_iris'] = [all_points[i] for i in self.RIGHT_IRIS]
            landmarks_dict['mouth'] = [all_points[i] for i in self.MOUTH_INDICES]
            landmarks_dict['nose_tip'] = all_points[self.NOSE_TIP]
            landmarks_dict['forehead'] = all_points[self.FOREHEAD]
            landmarks_dict['chin'] = all_points[self.CHIN]
        
        return landmarks_dict
    
    def draw_landmarks(self, frame, landmarks_dict):
        """Draw landmarks on frame"""
        if not landmarks_dict['face_detected']:
            return frame
        
        if landmarks_dict['all_landmarks'] is not None:
            for point in landmarks_dict['all_landmarks']:
                x = int(point[0])
                y = int(point[1])
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        
        # Draw left eye
        for point in landmarks_dict['left_eye']:
            x = int(point[0])
            y = int(point[1])
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
        
        # Draw right eye
        for point in landmarks_dict['right_eye']:
            x = int(point[0])
            y = int(point[1])
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
        
        # Draw left iris
        for point in landmarks_dict['left_iris']:
            x = int(point[0])
            y = int(point[1])
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
        
        # Draw right iris
        for point in landmarks_dict['right_iris']:
            x = int(point[0])
            y = int(point[1])
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
        
        # Draw mouth
        for point in landmarks_dict['mouth']:
            x = int(point[0])
            y = int(point[1])
            cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
        
        return frame