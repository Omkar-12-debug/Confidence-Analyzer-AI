"""
Video capture module for webcam and video file processing
Bhavesh - Visual Processing Lead
"""

import cv2
import numpy as np
import time
from datetime import datetime
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

class VideoCapture:
    def __init__(self, source=CAMERA_ID):
        """Initialize video capture"""
        self.source = source
        self.cap = None
        self.fps = FPS
        self.frame_count = 0
        self.is_recording = False
        self.video_writer = None
        
    def start_capture(self):
        """Start video capture from source"""
        self.cap = cv2.VideoCapture(self.source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, FPS)
        
        if not self.cap.isOpened():
            raise Exception("Could not open video source")
            
        print(f"[INFO] Video capture started from source: {self.source}")
        return True
    
    def get_frame(self):
        """Get single frame from video stream"""
        if self.cap is None:
            return None
            
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            return frame
        return None
    
    def start_recording(self, output_path=None):
        """Start recording video to file"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(OUTPUTS_DIR, 'processed_videos', f'recording_{timestamp}.avi')
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(
            output_path, 
            fourcc, 
            FPS, 
            (FRAME_WIDTH, FRAME_HEIGHT)
        )
        self.is_recording = True
        print(f"[INFO] Recording to: {output_path}")
        return output_path
    
    def write_frame(self, frame):
        """Write frame to recording file"""
        if self.is_recording and self.video_writer:
            self.video_writer.write(frame)
    
    def stop_recording(self):
        """Stop recording"""
        if self.video_writer:
            self.video_writer.release()
        self.is_recording = False
        print("[INFO] Recording stopped")
    
    def release(self):
        """Release video capture"""
        if self.cap:
            self.cap.release()
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()
        print("[INFO] Video capture released")
    
    def process_video_file(self, video_path, frame_processor, save_frames=False):
        """Process existing video file"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        frames_output_dir = os.path.join(FRAMES_DIR, video_name)
        if save_frames:
            os.makedirs(frames_output_dir, exist_ok=True)
        
        self.cap = cv2.VideoCapture(video_path)
        frame_count = 0
        processed_frames = []
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            processed_frame = frame_processor(frame, frame_count)
            processed_frames.append(processed_frame)
            
            if save_frames and frame_count % EXTRACT_EVERY_N_FRAMES == 0:
                frame_path = os.path.join(frames_output_dir, f'frame_{frame_count:06d}.jpg')
                cv2.imwrite(frame_path, frame)
            
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"[INFO] Processed {frame_count} frames")
        
        self.release()
        print(f"[INFO] Completed processing {frame_count} frames")
        return processed_frames

# This line is important - explicitly export the class
__all__ = ['VideoCapture']