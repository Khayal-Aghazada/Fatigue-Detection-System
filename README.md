# COMPUTER VISION SYSTEM FOR REAL-TIME DRIVER FATIGUE DETECTION

## Overview

This project implements a real-time fatigue detection system leveraging facial landmark analysis via **MediaPipe Face Mesh** combined with behavioral metrics such as blink rate, eye closure duration, head pose deviations, and yawning detection. The system captures video from a webcam, processes face landmarks, calculates relevant fatigue indicators, and alerts the user upon detecting signs of drowsiness or fatigue. Videos of testing are below in youtube link.

---

## Features

- **User-specific calibration**: A short calibration session records baseline eye aspect ratio (EAR), blink rate, and head pose, adapting thresholds to individual users for improved accuracy.
- **Blink detection**: Tracks eye closure duration and blink frequency to detect prolonged or excessive blinking, signs of fatigue.
- **Head pose monitoring**: Uses 3D head pose estimation to detect sustained head tilts or turns beyond configured thresholds.
- **Yawn detection**: Monitors mouth aspect ratio (MAR) to identify yawning events.
- **Real-time visual feedback**: Displays annotated webcam video with live fatigue indicators and alerts.
- **Audio alerts**: Plays alert sound when fatigue signs persist or intensify.
- **Modular and configurable**: Parameters can be fine-tuned via configuration and calibration data.

---

## System Architecture

### Core Components

- **CalibrationManager (`core/calibration.py`)**  
  Runs a calibration session (default 10-30 seconds) to collect baseline EAR, blink rate, and head pose angles. Computes user-specific thresholds for fatigue detection and saves them to a JSON profile.

- **FatigueDetection (`core/fatigue_detection.py`)**  
  Captures webcam video frames, processes them using MediaPipe Face Mesh to extract facial landmarks, computes behavioral metrics (EAR, MAR, head pose angles), and runs detectors to monitor fatigue signs. Alerts are triggered based on sustained threshold breaches.

- **Detectors (`core/detectors/`)**  
  - **BlinkDetector**: Detects blinks with exponential moving average smoothing and hysteresis logic.  
  - **AxisMotionDetector**: Detects sustained head pose deviations on pitch, yaw, and roll axes.  
  - **YawnDetector**: Detects yawns based on mouth aspect ratio thresholds and sustained duration.

- **Face Mesh (`core/face_mesh/`)**  
  Interfaces with MediaPipe’s Face Mesh model for efficient real-time face landmark detection and provides head pose estimation using OpenCV’s `solvePnP`.

- **Metrics (`core/metrics/`)**  
  Computes key metrics from landmarks:  
  - Eye Aspect Ratio (EAR) — ratio of vertical to horizontal eye distances.  
  - Mouth Aspect Ratio (MAR) — ratio indicating mouth openness, used for yawn detection.

- **Configuration (`config.py`)**  
  Stores constants and configurable parameters such as landmark indices, threshold values, EMA alpha, and alert timing.

---

## Installation

### Requirements

- Python 3.7+
- OpenCV (`opencv-python`)
- MediaPipe (`mediapipe`)
- NumPy

Install dependencies via pip:

```bash
pip install -r requirements.txt
````

---

## Usage

Run the system with:

```bash
python main.py
```

This will:

1. Start the calibration process (approx. 10-30 seconds), prompting the user to keep a neutral expression.
2. Save or load the calibration profile.
3. Begin continuous fatigue detection from the webcam.
4. Display real-time annotated video feed with fatigue metrics and alerts.
5. Play audio alerts upon detecting fatigue signs.
6. Exit cleanly when `q` key is pressed.

---

## Detailed Explanation of Key Algorithms

### Face Mesh & Landmark Detection

Uses **MediaPipe Face Mesh**, a lightweight neural network optimized for real-time facial landmark detection (468 landmarks), running on CPU or GPU.

### Eye Aspect Ratio (EAR)

Calculated as:

$$
EAR = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \times \|p_1 - p_4\|}
$$

where $p_i$ are 2D eye landmark coordinates.

Low EAR values indicate closed eyes, used to detect blinks and prolonged closure.

### Blink Detection

* Applies an **Exponential Moving Average (EMA)** to smooth EAR values.
* Uses hysteresis frames to avoid noise-induced false positives.
* Tracks blink duration to identify prolonged eye closure.
* Computes blink rate (blinks per minute) over a sliding window.

### Head Pose Estimation

* Uses selected 3D facial landmarks and OpenCV’s `solvePnP` to estimate pitch, yaw, and roll angles.
* Detects sustained head tilts beyond configurable thresholds for fatigue indication.

### Yawn Detection

* Calculates mouth aspect ratio (MAR) by measuring vertical to horizontal mouth opening.
* Detects yawns when MAR exceeds a threshold for a sustained period.

---

## Configuration Parameters

Parameters can be customized in `config.py` or dynamically updated from user profiles:

* Blink detection smoothing and thresholds (`EMA_ALPHA`, `THRESHOLD_DELTA`, etc.)
* Head pose angle thresholds (`PITCH_THRESHOLD`, `YAW_THRESHOLD`, `ROLL_THRESHOLD`)
* Timing parameters for sustained events and alert durations
* Landmark indices for eyes, mouth, and pose points

---

## Mobile Deployment Considerations

This system is designed primarily for desktop environments but can be ported to mobile with considerations:

* **MediaPipe Face Mesh** offers official mobile SDKs optimized for Android and iOS, supporting GPU acceleration.
* Video capture and UI rendering must be adapted to platform-specific APIs.
* Real-time processing demands can be reduced by tuning resolution and frame rates.
* Audio playback code (`winsound`) requires replacement with cross-platform APIs.
* Performance profiling and optimization are essential to maintain smooth user experience.
* Battery usage and thermal constraints must be addressed.

---

## Project Structure

```
core/
├── calibration.py         # User-specific calibration routine
├── fatigue_detection.py   # Main fatigue detection loop and logic
├── detectors/
│   ├── blink_detector.py  # Blink detection logic
│   ├── motion_detector.py # Head motion detection logic
│   └── yawn_detector.py   # Yawn detection logic
├── face_mesh/
│   ├── face_mesh_detector.py # MediaPipe Face Mesh interface
│   └── head_pose_estimator.py # Head pose calculation
├── metrics/
│   ├── eye_aspect_ratio.py   # EAR calculation
│   └── mouth_aspect_ratio.py # MAR calculation
config.py               # Configuration and constants
main.py                 # Entry point for calibration + detection
alert.wav               # Audio alert file
requirements.txt        # Python dependencies
README.md               # This documentation
```

---

## How to Contribute

Feel free to submit issues and pull requests. Suggestions for improvements include:

* Adding additional fatigue indicators (e.g., facial expression analysis)
* Enhancing mobile compatibility and creating native apps
* Improving UI and alerting mechanisms
* Adding data logging and reporting

---

Testing video links:
https://youtu.be/edc16mRqpAI
https://youtu.be/ztCU1SqB2EM
https://youtu.be/Gpzoid237Pg

---

## Contact

For questions or support, contact: khayal.aghazada.x@gmail.com or https://github.com/Khayal-Aghazada

---

*This system leverages MediaPipe by Google and OpenCV libraries, both open-source.*

---
