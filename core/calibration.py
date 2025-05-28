# core/calibration.py

import cv2, time, json, os
from face_mesh.face_mesh_detector import FaceMeshDetector
from metrics.eye_aspect_ratio   import compute_ear
from face_mesh.head_pose_estimator import compute_head_pose
from config                     import (
    LEFT_EYE_IDX, RIGHT_EYE_IDX,
    LANDMARK_INDICES,
)

PROFILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'user_profile.json')

class CalibrationManager:
    """
    Runs a brief session to record baseline EAR blink rate and head-pose,
    then computes user-specific parameters and saves them.
    """

    def __init__(self, duration_s=10):
        self.duration = duration_s
        self.mesh     = FaceMeshDetector()
        self.cap      = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera for calibration")

    def run(self):
        start = time.time()
        ears, pitch_devs = [], []
        blink_times = []
        last_ear = None
        closed = False

        while time.time() - start < self.duration:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame,1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h,w = frame.shape[:2]
            res = self.mesh.process_frame(rgb)
            if not res.multi_face_landmarks:
                continue

            lm = res.multi_face_landmarks[0].landmark
            ear = (compute_ear(lm, LEFT_EYE_IDX, w,h) +
                   compute_ear(lm, RIGHT_EYE_IDX,w,h)) / 2
            ears.append(ear)

            thresh = min(ears) + (max(ears)-min(ears))*0.5
            if ear < thresh and not closed:
                blink_times.append(time.time())
                closed = True
            elif ear >= thresh:
                closed = False

            pitch,_,_ = compute_head_pose(lm, w,h)
            if pitch is not None:
                pitch_devs.append(pitch)

        self.cap.release()
        self.mesh.close()

        avg_ear = sum(ears)/len(ears)
        ear_std = (sum((e-avg_ear)**2 for e in ears)/len(ears))**0.5
        blink_rate = len(blink_times)/(self.duration/60)

        avg_pitch = sum(pitch_devs)/len(pitch_devs)
        pitch_std = (sum((p-avg_pitch)**2 for p in pitch_devs)/len(pitch_devs))**0.5

        user_profile = {
            "blink": {
                "alpha": 0.1,
                "threshold_delta": ear_std * 0.5,
                "hysteresis_frames": int(ear_std * 30) or 3
            },
            "head_pose": {
                "pitch_threshold": avg_pitch + 2*pitch_std,
                "sustain_time": 3.0
            },
            "baseline": {
                "avg_ear": avg_ear,
                "avg_pitch": avg_pitch,
                "blink_rate": blink_rate
            }
        }

        with open(PROFILE_PATH, 'w') as f:
            json.dump(user_profile, f, indent=2)

        return user_profile

    def load_or_run(self):
        if os.path.exists(PROFILE_PATH):
            with open(PROFILE_PATH) as f:
                return json.load(f)
        return self.run()
