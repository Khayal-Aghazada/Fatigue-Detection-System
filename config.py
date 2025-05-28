# config.py
import mediapipe as mp
import numpy as np
import json, os

MP_FACE_MESH_CONFIG = {
    "static_image_mode": False,
    "max_num_faces": 1,
    "refine_landmarks": True,
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5
}

mp_drawing    = mp.solutions.drawing_utils
DRAWING_SPEC  = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

LEFT_EYE_IDX     = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX    = [263, 387, 385, 362, 380, 373]
EMA_ALPHA        = 0.1
THRESHOLD_DELTA  = 0.02
HYSTERESIS_FRAMES= 3
MIN_BLINK_DURATION = 0.2
BLINK_RATE_WINDOW   = 60

MOUTH_UPPER_IDX = 13
MOUTH_LOWER_IDX = 14
MOUTH_LEFT_IDX  = 78
MOUTH_RIGHT_IDX = 308
MOUTH_AR_THRESH = 0.6

LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]
MODEL_POINTS     = np.array([
    [0.0,   0.0,    0.0],
    [0.0, -63.6, -12.5],
    [-43.3, 32.7, -26.0],
    [43.3,  32.7, -26.0],
    [-28.9,-28.9, -24.1],
    [28.9, -28.9, -24.1]
], dtype=np.float32)

PITCH_THRESHOLD            = 20
YAW_THRESHOLD              = 20
ROLL_THRESHOLD             = 20
SUSTAINED_TIME_THRESHOLD   = 3.0
ALERT_DISPLAY_DURATION     = 1.0


PROFILE_PATH = os.path.join(os.path.dirname(__file__), 'user_profile.json')

def apply_user_profile(profile):

    bp = profile.get("blink", {})
    globals().update({
        'EMA_ALPHA': bp.get('alpha', EMA_ALPHA),
        'THRESHOLD_DELTA': bp.get('threshold_delta', THRESHOLD_DELTA),
        'HYSTERESIS_FRAMES': bp.get('hysteresis_frames', HYSTERESIS_FRAMES),
    })

    hp = profile.get("head_pose", {})
    globals().update({
        'PITCH_THRESHOLD': hp.get('pitch_threshold', PITCH_THRESHOLD),
        'SUSTAINED_TIME_THRESHOLD': hp.get('sustain_time', SUSTAINED_TIME_THRESHOLD),
    })