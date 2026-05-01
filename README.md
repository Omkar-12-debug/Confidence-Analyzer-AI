# 🎯 Confidence Analyzer AI

### Multimodal Confidence & Stress Analysis using Speech and Facial Behavior

---

## 📌 Overview

**Confidence Analyzer AI** is a desktop-based multimodal system that evaluates a user's confidence level by analyzing both **voice characteristics** and **facial behavior** during a recorded response.

The system integrates:

* 🎤 **Speech analysis** (pitch, energy, pauses, speech rate)
* 🎥 **Facial behavior analysis** (eye contact, blink rate, emotional stability)
* 🤖 **Fusion model** to produce an overall confidence score

The application is packaged as a **cross-platform desktop app using Tauri**, enabling real-world usability without requiring a browser or manual setup.

---

## 🚀 Key Features

* ✅ Multimodal AI-based confidence evaluation
* ✅ Voice-only fallback when face is not detected
* ✅ Intelligent validation (rejects silence/no speech)
* ✅ Facial behavior insights (non-intrusive, no fake scoring)
* ✅ Real-time recording and analysis
* ✅ Clean, interactive dashboard UI
* ✅ Desktop deployment (no server dependency for users)

---

## 🧠 System Architecture

```
User Recording (Video + Audio)
            │
            ▼
    ┌──────────────────────┐
    │  Frontend (React +   │
    │  Tauri Desktop UI)   │
    └──────────────────────┘
            │
            ▼
    ┌──────────────────────┐
    │ Backend (Python API) │
    │  - Audio Processing  │
    │  - Visual Processing │
    │  - Fusion Model      │
    └──────────────────────┘
            │
            ▼
     Confidence Score + Insights
```

---

## 🔍 Analysis Modules

### 🎤 Audio Analysis

Extracts key speech features:

* Pitch (mean, variation)
* Energy levels
* Speech rate
* Pause ratio
* MFCC-based characteristics

👉 Used to determine **vocal confidence and fluency**

---

### 🎥 Facial Analysis

Extracts behavioral cues:

* Eye contact percentage
* Blink rate
* Head movement
* Emotion stability

👉 Used to assess **engagement and composure**

---

### 🔗 Fusion Logic

The system intelligently combines modalities:

| Scenario       | Output                           |
| -------------- | -------------------------------- |
| Audio + Facial | Full Multimodal Confidence       |
| Audio only     | Voice-only Confidence            |
| Facial only    | Incomplete (no confidence score) |
| No valid input | Invalid recording                |

👉 Speech is treated as **mandatory** for final confidence evaluation.

---

## ⚙️ Technology Stack

### Frontend

* React (TypeScript)
* Tailwind CSS + shadcn UI
* Vite
* Tauri (Desktop shell)

### Backend

* Python (Flask)
* Librosa (audio processing)
* OpenCV (facial detection)
* Scikit-learn (ML models)

### Additional Tools

* FFmpeg (audio extraction)
* PyInstaller (backend packaging)

---

## 📥 Download & Installation

👉 Download the application from:
**[GitHub Releases Link]**

### Steps:

1. Run the installer (`.exe`)
2. Allow **camera and microphone permissions**
3. Click **Start Recording**
4. Speak clearly for accurate analysis

---

## ⚠️ Important Notes

* Speech input is required for full confidence evaluation
* Facial cues alone do not produce final confidence scores
* Ensure good lighting and clear audio for best results

---

## 🧪 Testing Scenarios

The system supports:

* ✔ Full video + audio analysis
* ✔ Audio-only evaluation
* ✔ Facial-only behavioral insights (no scoring)
* ✔ Silent/no-input rejection

---

## 📊 Future Enhancements

* Real-time (live) confidence tracking
* Cloud deployment with user analytics
* Improved deep learning-based emotion detection
* Mobile version of the application

---

## 👨‍💻 Authors

* Omkar Waghade
* Parth Moghe
* Heet Patel
* Bhavesh Mundake

---

## 📄 License

This project is for academic and research purposes.

---

## ⭐ Acknowledgment

This project was developed as part of an AI/ML-based system to explore multimodal human behavior analysis for confidence assessment in real-world scenarios such as interviews and presentations.

---
