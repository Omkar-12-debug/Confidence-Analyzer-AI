def calculate_confidence_score(blink_rate, eye_contact_percentage, head_movement_frequency, emotion_stability, emotion_confidence):
    """
    Calculates a heuristic-based preliminary confidence score from facial features.
    
    Args:
        blink_rate (float): Blinks per minute (lower is generally more confident, up to a point).
        eye_contact_percentage (float): Percentage of time maintaining eye contact (0-100).
        head_movement_frequency (float): Frequency of head movements (excessive is lower confidence).
        emotion_stability (float): Stability of emotion over time (0.0 - 1.0).
        emotion_confidence (float): Confidence of the emotion classification itself (0.0 - 1.0).
        
    Returns:
        float: A confidence score between 0.0 and 1.0.
    """
    
    # 1. Normalize inputs to a 0.0 - 1.0 scale
    # Heuristics:
    # - Ideal blink rate is around 15-20. Very high (>30) or very low (<10) might indicate stress/lack of confidence.
    #   We'll penalize rates above 25.
    norm_blink = max(0.0, 1.0 - (max(0, blink_rate - 20) / 30.0))  # Decays as blink rate goes > 20
    
    # - Eye contact directly correlates to confidence (ideal is high, e.g., > 70%).
    norm_eye_contact = min(1.0, max(0.0, eye_contact_percentage / 100.0))
    
    # - Excessive head movement correlates with nervousness. Let's say > 20 is excessive.
    norm_head_movement = max(0.0, 1.0 - (head_movement_frequency / 40.0)) 
    
    # Emotion stability and confidence are already 0-1
    norm_emotion_stability = min(1.0, max(0.0, emotion_stability))
    norm_emotion_confidence = min(1.0, max(0.0, emotion_confidence))
    
    # 2. Weighted features
    # Let's assign weights based on expected importance
    weights = {
        'eye_contact': 0.35,
        'emotion_stability': 0.25,
        'head_movement': 0.20,
        'blink_rate': 0.10,
        'emotion_confidence': 0.10
    }
    
    # 3. Calculate score
    score = (
        norm_eye_contact * weights['eye_contact'] +
        norm_emotion_stability * weights['emotion_stability'] +
        norm_head_movement * weights['head_movement'] +
        norm_blink * weights['blink_rate'] +
        norm_emotion_confidence * weights['emotion_confidence']
    )
    
    # Ensure the final score is strictly between 0 and 1
    return max(0.0, min(1.0, score))

def main():
    # Sample confident person
    sample_confident = {
        'blink_rate': 18,
        'eye_contact_percentage': 85,
        'head_movement_frequency': 12,
        'emotion_stability': 0.9,
        'emotion_confidence': 0.95
    }
    
    score_confident = calculate_confidence_score(**sample_confident)
    print(f"Sample Confident Person Features: {sample_confident}")
    print(f"Calculated Confidence Score: {score_confident:.2f}\n")

    # Sample nervous person
    sample_nervous = {
        'blink_rate': 35,
        'eye_contact_percentage': 40,
        'head_movement_frequency': 32,
        'emotion_stability': 0.4,
        'emotion_confidence': 0.6
    }
    
    score_nervous = calculate_confidence_score(**sample_nervous)
    print(f"Sample Nervous Person Features: {sample_nervous}")
    print(f"Calculated Confidence Score: {score_nervous:.2f}")

if __name__ == "__main__":
    main()
