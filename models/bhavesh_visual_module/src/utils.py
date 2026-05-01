"""
Utility functions for visual processing module
Bhavesh - Visual Processing Lead
"""

import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from datetime import datetime

def load_video_list(directory):
    """Load all video files from directory"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    videos = []
    
    for file in os.listdir(directory):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            videos.append(os.path.join(directory, file))
    
    return videos

def create_visualization(metrics_history, output_path=None):
    """Create visualization plots of metrics over time"""
    if not metrics_history:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Extract data
    frames = [m['frame_number'] for m in metrics_history]
    blink_rates = [m.get('blink_rate', 0) for m in metrics_history]
    eye_contact = [m.get('eye_contact_percentage', 0) for m in metrics_history]
    head_movement = [m.get('head_movement_frequency', 0) for m in metrics_history]
    emotion_stability = [m.get('emotion_stability', 1) for m in metrics_history]
    
    # Plot 1: Blink Rate
    axes[0, 0].plot(frames, blink_rates)
    axes[0, 0].set_title('Blink Rate Over Time')
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylabel('Blinks/Minute')
    
    # Plot 2: Eye Contact
    axes[0, 1].plot(frames, eye_contact)
    axes[0, 1].set_title('Eye Contact Percentage')
    axes[0, 1].set_xlabel('Frame')
    axes[0, 1].set_ylabel('Percentage')
    
    # Plot 3: Head Movement
    axes[0, 2].plot(frames, head_movement)
    axes[0, 2].set_title('Head Movement Frequency')
    axes[0, 2].set_xlabel('Frame')
    axes[0, 2].set_ylabel('Movements/Minute')
    
    # Plot 4: Emotion Stability
    axes[1, 0].plot(frames, emotion_stability)
    axes[1, 0].set_title('Emotion Stability')
    axes[1, 0].set_xlabel('Frame')
    axes[1, 0].set_ylabel('Stability Score')
    
    # Plot 5: Emotion Distribution
    emotions = [m.get('facial_emotion', 'Neutral') for m in metrics_history]
    emotion_counts = {}
    for e in emotions:
        emotion_counts[e] = emotion_counts.get(e, 0) + 1
    
    axes[1, 1].pie(emotion_counts.values(), labels=emotion_counts.keys(), autopct='%1.1f%%')
    axes[1, 1].set_title('Emotion Distribution')
    
    # Plot 6: Summary Stats
    stats_text = f"Total Frames: {len(frames)}\n"
    stats_text += f"Avg Blink Rate: {np.mean(blink_rates):.2f}\n"
    stats_text += f"Avg Eye Contact: {np.mean(eye_contact):.2f}%\n"
    stats_text += f"Avg Head Movement: {np.mean(head_movement):.2f}\n"
    stats_text += f"Avg Emotion Stability: {np.mean(emotion_stability):.2f}"
    
    axes[1, 2].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('Summary Statistics')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"[INFO] Visualization saved to {output_path}")
    
    plt.show()

def export_for_fusion(features, output_path=None):
    """Export features in format expected by fusion module"""
    if output_path is None:
        output_path = os.path.join('outputs', 'fusion_input.json')
    
    fusion_format = {
        'visual_features': features,
        'timestamp': datetime.now().isoformat(),
        'module': 'bhavesh_visual'
    }
    
    with open(output_path, 'w') as f:
        json.dump(fusion_format, f, indent=2)
    
    return output_path

def calculate_confidence_score(metrics):
    """Calculate preliminary confidence score from visual metrics"""
    score = 0.0
    weights = {
        'eye_contact': 0.3,
        'emotion_stability': 0.25,
        'blink_rate': 0.2,
        'head_movement': 0.15,
        'emotion_positive': 0.1
    }
    
    # Eye contact (higher is better)
    eye_contact = metrics.get('eye_contact_percentage', 0) / 100
    score += weights['eye_contact'] * eye_contact
    
    # Emotion stability (higher is better)
    stability = metrics.get('emotion_stability', 1.0)
    score += weights['emotion_stability'] * stability
    
    # Blink rate (moderate is better - not too many, not too few)
    blink_rate = metrics.get('blink_rate', 15)
    blink_score = 1.0 - min(abs(blink_rate - 15) / 30, 1.0)
    score += weights['blink_rate'] * blink_score
    
    # Head movement (less is better for confidence)
    head_move = metrics.get('head_movement_frequency', 10)
    head_score = 1.0 - min(head_move / 30, 1.0)
    score += weights['head_movement'] * head_score
    
    # Emotion positivity
    positive_emotions = ['Happy', 'Neutral']
    emotion = metrics.get('facial_emotion', 'Neutral')
    emotion_score = 1.0 if emotion in positive_emotions else 0.5
    score += weights['emotion_positive'] * emotion_score
    
    return score