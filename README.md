# Gesture Detection Text-to-Speech System

## Overview

This project is a real-time American Sign Language (ASL) gesture recognition system that converts hand gestures into text and speech using Computer Vision, Deep Learning, and Text-to-Speech (TTS).

The application captures live webcam input, detects hand gestures, predicts ASL alphabet characters using a trained CNN model, forms words/sentences, and speaks the output aloud.

This is not a basic demo-level project. The system combines:

* Real-time hand tracking
* Gesture recognition using Deep Learning
* GUI-based interaction
* Word suggestion and spell correction
* Text-to-Speech output
* Dataset collection utilities

---

# Features

## Real-Time Gesture Recognition

* Detects hand gestures directly from webcam feed
* Supports ASL alphabet prediction
* Uses hand skeleton tracking for accurate recognition

## Deep Learning Model

* CNN-based gesture classification
* Pre-trained `.h5` model for prediction
* Multi-stage prediction logic for improved accuracy

## Smart Sentence Formation

* Converts predicted characters into words/sentences
* Includes:

  * Space handling
  * Backspace support
  * Character confirmation

## Spell Suggestions

* Integrated spell-checking system
* Suggests possible words dynamically
* Helps reduce prediction errors

## Text-to-Speech

* Speaks generated sentences using TTS engine
* Useful for communication assistance

## GUI Interface

* Built using Tkinter
* Displays:

  * Live webcam feed
  * Hand skeleton
  * Predicted character
  * Sentence output
  * Word suggestions

## Dataset Collection Scripts

* Includes utilities for:

  * Binary image dataset collection
  * Final dataset generation

---

# Project Structure

```bash
Gesture_detection_Text_to_Speech-main/
│
├── AtoZ_3.1/                     # Dataset folders (A-Z images)
├── Gesture detection/
│   └── detection.py              # Detection utilities
│
├── HandTrackingModule.py         # Custom hand tracking module
├── data_collection_binary.py     # Binary dataset collection
├── data_collection_final.py      # Final dataset creation
├── final_pred.py                 # Main application
├── prediction_wo_gui.py          # Prediction without GUI
├── cnn8grps_rad1_model.h5        # Trained CNN model
├── README.md
└── .gitignore
```

---
# ScreenShot


# Technologies Used

## Programming Language

* Python

## Computer Vision

* OpenCV
* MediaPipe
* CVZone

## Deep Learning

* TensorFlow / Keras

## GUI

* Tkinter

## Text-to-Speech

* pyttsx3

## NLP Utilities

* pyspellchecker

---

# System Workflow

```text
Webcam Input
      ↓
Hand Detection
      ↓
Hand Landmark Extraction
      ↓
CNN Model Prediction
      ↓
Character Recognition
      ↓
Sentence Formation
      ↓
Spell Suggestions
      ↓
Text-to-Speech Output
```

---

# Installation

## 1. Clone the Repository

```bash
git clone <repository-url>
cd Gesture_detection_Text_to_Speech-main
```

---

## 2. Create Virtual Environment (Recommended)

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install opencv-python
pip install numpy
pip install mediapipe
pip install cvzone
pip install tensorflow
pip install keras
pip install pillow
pip install pyttsx3
pip install pyspellchecker
```

Or create a `requirements.txt` file and install everything together.

---

# Running the Project

## Run Main GUI Application

```bash
python final_pred.py
```

---

## Run Prediction Without GUI

```bash
python prediction_wo_gui.py
```

---

## Dataset Collection

### Binary Dataset Collection

```bash
python data_collection_binary.py
```

### Final Dataset Collection

```bash
python data_collection_final.py
```

---

# Controls

| Key       | Function                   |
| --------- | -------------------------- |
| Enter     | Confirm detected character |
| Space     | Add space                  |
| Backspace | Delete previous character  |

---

# Model Details

The system uses a Convolutional Neural Network (CNN) trained on ASL hand gesture images.

## Model Responsibilities

* Feature extraction from hand images
* Gesture classification
* Alphabet prediction

## Input

* Hand gesture image/frame

## Output

* Predicted ASL character

---

# Applications

## Communication Assistance

Helps speech or hearing-impaired individuals communicate more effectively.

## Human-Computer Interaction

Can be extended into gesture-controlled systems.

## Smart Assistive Systems

Useful in accessibility-focused AI systems.

## Educational Systems

Can be used for ASL learning platforms.

---

# Future Improvements

* Add support for complete ASL words and phrases
* Improve prediction accuracy with larger datasets
* Deploy as a desktop executable
* Add multilingual speech support
* Convert into mobile application
* Add cloud-based model serving
* Implement transformer-based gesture recognition

---

# Challenges Faced

* Real-time gesture stability
* Background noise in webcam frames
* Similar-looking gesture classification
* Lighting condition variations
* Maintaining low prediction latency

---

# Conclusion

This project demonstrates the practical integration of:

* Artificial Intelligence
* Deep Learning
* Computer Vision
* Human-Computer Interaction
* Assistive Technology

The system provides a strong foundation for building advanced gesture-based communication systems and accessibility-focused AI applications.

---

# Author

Developed as an AI/ML and Computer Vision project focused on real-time ASL gesture recognition and speech generation.

---

# License

This project is intended for educational and research purposes.

