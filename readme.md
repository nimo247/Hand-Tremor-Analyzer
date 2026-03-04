---
title: Hand Tremor Analyzer
emoji: 🖐
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
---

# Hand Tremor Analyzer

Real-time Parkinson's tremor detection using Computer Vision.

## How to Use
1. Allow camera access
2. Hold hand in front of camera
3. Wait 15 frames for calibration
4. See live tremor analysis

## Tech Stack
- Python
- MediaPipe
- OpenCV
- Streamlit
- SciPy (FFT)
```

This tells Hugging Face to use Streamlit automatically.

---

## Step 5 — Create requirements.txt

Make sure your `requirements.txt` looks exactly like this:
```
mediapipe
opencv-python-headless
numpy
scipy
streamlit
reportlab
matplotlib
pandas
```

---

## Step 6 — Create the GitHub Actions Workflow File

In your project folder create these folders and file:
```
your_project/
├── .github/
│   └── workflows/
│       └── deploy.yml    ← create this file
├── app.py
├── tremor_analyzer.py
├── handtrackingmodule.py
├── requirements.txt
└── README.md