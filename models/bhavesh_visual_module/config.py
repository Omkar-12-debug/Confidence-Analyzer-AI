"""
Configuration file for Visual Processing Module
Bhavesh - Visual Processing Lead
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_VIDEOS_DIR = os.path.join(DATA_DIR, 'raw_videos')
FRAMES_DIR = os.path.join(DATA_DIR, 'frames')
FEATURES_DIR = os.path.join(DATA_DIR, 'processed_features')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')

# Video Capture Settings
CAMERA_ID = 0  # Default webcam
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30
RECORDING_DURATION = 60  # seconds

# Face Detection Settings
FACE_DETECTION_CONFIDENCE = 0.5
USE_MEDIAPIPE = True

# Emotion Recognition Settings
EMOTION_MODEL_PATH = os.path.join(MODELS_DIR, 'emotion_model', 'fer2013_mini_XCEPTION.102-0.66.hdf5')
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Behavior Metrics Settings
BLINK_THRESHOLD = 0.2
EYE_CONTACT_THRESHOLD = 0.3
HEAD_MOVEMENT_SMOOTHING = 5

# Feature Extraction Settings
EXTRACT_EVERY_N_FRAMES = 5
SAVE_VIDEO_OUTPUT = True
SHOW_VISUALIZATION = True