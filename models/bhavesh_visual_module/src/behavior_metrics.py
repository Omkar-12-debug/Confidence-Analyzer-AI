"""
Behavior metrics calculation module
Bhavesh - Visual Processing Lead
"""

import numpy as np
from collections import deque
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

class BehaviorMetrics:
    def __init__(self, buffer_size=150):
        """Initialize behavior metrics calculator"""
        self.buffer_size = buffer_size
        
        # Metrics history
        self.ear_history = deque(maxlen=buffer_size)
        self.gaze_history = deque(maxlen=buffer_size)
        self.head_position_history = deque(maxlen=buffer_size)
        self.emotion_history = deque(maxlen=buffer_size)
        
        # State for blink detection
        self.is_eye_closed = False
        self.closed_frame_count = 0
        self.blink_count = 0
        
        # State for eye contact
        self.eye_contact_frames = 0
        
        # State for head movement
        self.head_movement_count = 0
        self.prev_head_pos = None
        
        # Timers / Counters
        self.start_time = time.time()
        self.total_frames = 0
        
        print("[INFO] Behavior metrics initialized")
    
    def update_metrics(self, landmarks_dict, emotion=None, emotion_conf=None):
        """Update all metrics with new frame data"""
        self.total_frames += 1
        
        if not landmarks_dict['face_detected']:
            return
        
        # Calculate EAR
        if landmarks_dict['left_eye'] and landmarks_dict['right_eye']:
            left_ear = self._calculate_ear(landmarks_dict['left_eye'])
            right_ear = self._calculate_ear(landmarks_dict['right_eye'])
            avg_ear = (left_ear + right_ear) / 2.0
            self.ear_history.append(avg_ear)
            self._detect_blink(avg_ear)
        
        # Calculate eye contact (replace gaze with head orientation)
        if landmarks_dict.get('nose_tip') and landmarks_dict.get('left_eye') and landmarks_dict.get('right_eye'):
            orientation = self._calculate_head_orientation(
                landmarks_dict['nose_tip'],
                landmarks_dict['left_eye'],
                landmarks_dict['right_eye']
            )
            self.gaze_history.append(orientation)
            if orientation == 'center':
                self.eye_contact_frames += 1
        
        # Track head movement
        if landmarks_dict['nose_tip']:
            self._track_head_movement(landmarks_dict['nose_tip'])
        
        # Store emotion
        if emotion and emotion_conf:
            self.emotion_history.append((emotion, emotion_conf))
    
    def _calculate_ear(self, eye_landmarks):
        """Calculate Eye Aspect Ratio"""
        if len(eye_landmarks) < 6:
            return 0.0
        
        eye_points = np.array([[p[0], p[1]] for p in eye_landmarks])
        
        # Vertical distances
        vert_dist1 = np.linalg.norm(eye_points[1] - eye_points[5])
        vert_dist2 = np.linalg.norm(eye_points[2] - eye_points[4])
        
        # Horizontal distance
        horz_dist = np.linalg.norm(eye_points[0] - eye_points[3])
        
        # Calculate EAR
        ear = (vert_dist1 + vert_dist2) / (2.0 * horz_dist + 1e-6)
        
        return ear
    
    def _detect_blink(self, ear):
        """Detect blink using stateful EAR threshold and consecutive frames"""
        EAR_THRESHOLD = 0.22
        CONSECUTIVE_FRAMES = 2
        
        if ear < EAR_THRESHOLD:
            self.closed_frame_count += 1
            self.is_eye_closed = True
        else:
            if self.is_eye_closed and self.closed_frame_count >= CONSECUTIVE_FRAMES:
                self.blink_count += 1
            self.is_eye_closed = False
            self.closed_frame_count = 0
    
    def _calculate_head_orientation(self, nose_tip, left_eye, right_eye):
        """Approximate head orientation using nose displacement relative to eyes"""
        if not nose_tip or not left_eye or not right_eye:
            return 'unknown'
        
        nose_x = nose_tip[0]
        left_x = np.mean([p[0] for p in left_eye])
        right_x = np.mean([p[0] for p in right_eye])
        
        eye_center_x = (left_x + right_x) / 2.0
        eye_width = abs(right_x - left_x)
        
        if eye_width == 0:
            return 'unknown'
            
        offset = (nose_x - eye_center_x) / eye_width
        
        if abs(offset) < 0.05: # Tightened to capture sub-frame micro head drift
            return 'center'
        else:
            return 'away'
    
    def _track_head_movement(self, nose_tip):
        """Track head movement using nose tip displacement threshold"""
        current_pos = np.array([nose_tip[0], nose_tip[1]])
        MOVE_THRESHOLD = 5.0
        
        if self.prev_head_pos is not None:
            distance = np.linalg.norm(current_pos - self.prev_head_pos)
            if distance > MOVE_THRESHOLD:
                self.head_movement_count += 1
        
        self.prev_head_pos = current_pos
        self.head_position_history.append(current_pos)
    
    def get_blink_rate(self):
        """Calculate blinks per second (normalized)"""
        duration_sec = self.total_frames / 30.0
        if duration_sec > 0:
            return self.blink_count / duration_sec
        return 0.0
    
    def get_eye_contact_percentage(self):
        """Calculate percentage of frames with eye contact"""
        total_frames_with_face = len(self.gaze_history)
        if total_frames_with_face > 0:
            return (self.eye_contact_frames / total_frames_with_face) * 100
        return 0
    
    def get_head_movement_frequency(self):
        """Calculate head movements per second, scaled to 0-10"""
        duration_sec = self.total_frames / 30.0
        if duration_sec > 0:
            freq = self.head_movement_count / duration_sec
            # Scale range 0-10 (clamped)
            return min(max(freq, 0), 10.0)
        return 0.0
    
    def get_emotion_stability(self):
        """Calculate emotion stability score"""
        if len(self.emotion_history) < 10:
            return 1.0
        
        emotions = [e[0] for e in self.emotion_history]
        changes = sum(1 for i in range(1, len(emotions)) if emotions[i] != emotions[i-1])
        
        stability = 1.0 - (changes / (len(emotions) - 1))
        return stability
    
    def get_dominant_emotion(self):
        """Get most frequent emotion in history"""
        if len(self.emotion_history) == 0:
            return 'Neutral', 0.0
        
        emotions = [e[0] for e in self.emotion_history]
        confidences = [e[1] for e in self.emotion_history]
        
        from collections import Counter
        emotion_counts = Counter(emotions)
        dominant = emotion_counts.most_common(1)[0][0]
        
        avg_confidence = np.mean([c for e, c in self.emotion_history if e == dominant])
        
        return dominant, float(avg_confidence)
    
    def get_all_metrics(self):
        """Get all calculated metrics"""
        dominant_emotion, emotion_conf = self.get_dominant_emotion()
        
        return {
            'facial_emotion': dominant_emotion,
            'emotion_confidence': emotion_conf,
            'emotion_stability': self.get_emotion_stability(),
            'blink_rate': self.get_blink_rate(),
            'eye_contact_percentage': self.get_eye_contact_percentage(),
            'head_movement_frequency': self.get_head_movement_frequency(),
            'total_blinks': self.blink_count,
            'total_frames': self.total_frames,
            'analysis_duration': self.total_frames / 30.0
        }
    
    def reset(self):
        """Reset all metrics"""
        self.__init__()