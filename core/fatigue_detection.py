import os
import cv2
import time
import winsound
import threading
import wave
import contextlib
import mediapipe as mp

from config import (
    mp_drawing,
    DRAWING_SPEC,
    LEFT_EYE_IDX,
    RIGHT_EYE_IDX,
    EMA_ALPHA,
    THRESHOLD_DELTA,
    HYSTERESIS_FRAMES,
    MIN_BLINK_DURATION,
    BLINK_RATE_WINDOW,
    LANDMARK_INDICES,
    MODEL_POINTS,
    PITCH_THRESHOLD,
    YAW_THRESHOLD,
    ROLL_THRESHOLD,
    SUSTAINED_TIME_THRESHOLD,
    ALERT_DISPLAY_DURATION,
    MOUTH_AR_THRESH,
)

from face_mesh.face_mesh_detector import FaceMeshDetector
from face_mesh.head_pose_estimator import compute_head_pose

from metrics.eye_aspect_ratio   import compute_ear
from metrics.mouth_aspect_ratio import compute_mar

from detectors.blink_detector  import BlinkDetector
from detectors.motion_detector import AxisMotionDetector
from detectors.yawn_detector   import YawnDetector


class FatigueDetection:
    def __init__(self):
        # Mediapipe face mesh
        self.mesh      = FaceMeshDetector()

        # Stateful detectors
        self.blink_det = BlinkDetector()
        self.yawn_det  = YawnDetector()
        self.pitch_det = AxisMotionDetector(PITCH_THRESHOLD)
        self.yaw_det   = AxisMotionDetector(YAW_THRESHOLD)
        self.roll_det  = AxisMotionDetector(ROLL_THRESHOLD)

        # EMA baseline for pitch
        self.pitch_base = None

        # Video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")

        # Alert sound path and cooldown setup
        sound_path = os.path.join(os.path.dirname(__file__), '..', 'alert.wav')
        self.alert_path = sound_path
        self.extra_cd   = 1.5  # extra pause after playback
        self.last_sound = 0.0
        # total cooldown = file duration + extra
        self.sound_cd   = self._get_wav_duration(self.alert_path) + self.extra_cd

        # High-rate blink alert cooldown (unchanged)
        self.high_rate_cd   = 10.0
        self.last_high_rate = 0.0

        # —— Scoring window (15 s) ——
        self.window_start   = time.time()
        self.window_score   = 0.0
        self.window_duration = 15.0

    def _get_wav_duration(self, path):
        """Return WAV duration in seconds."""
        with contextlib.closing(wave.open(path,'r')) as wf:
            frames = wf.getnframes()
            rate   = wf.getframerate()
            return frames / float(rate)

    def _play_sound(self):
        """Play alert.wav fully, then enforce extra cooldown."""
        now = time.time()
        # suppress if within cooldown period
        if now - self.last_sound < self.sound_cd:
            return

        def _worker():
            # blocking play until file ends (omitting SND_ASYNC for sync playback)
            winsound.PlaySound(self.alert_path,
                               winsound.SND_FILENAME)
            # extra wait after playback
            time.sleep(self.extra_cd)

        # start playback thread
        threading.Thread(target=_worker, daemon=True).start()
        # record start time to prevent overlap
        self.last_sound = now

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w  = frame.shape[:2]

            results = self.mesh.process_frame(rgb)
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark

                # --- 1) Blink Detection ---
                ear = (
                    compute_ear(lm, LEFT_EYE_IDX,  w, h) +
                    compute_ear(lm, RIGHT_EYE_IDX, w, h)
                ) / 2.0
                thresh, rate, msgs = self.blink_det.update(ear)

                cv2.putText(frame, f"EAR: {ear:.3f}",      (30,  30),
                            cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,0), 2)
                cv2.putText(frame, f"Thr: {thresh:.3f}",   (30,  60),
                            cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
                for i, m in enumerate(msgs):
                    cv2.putText(frame, m, (30, 90 + i*30),
                                cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,255), 2)
                cv2.putText(frame, f"Rate: {rate:.1f} blinks/min",
                            (30,150), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,0,0), 2)

                if rate > 19:
                    cv2.putText(frame, "Fatigue Alert: High Blink Rate!",
                                (30,180), cv2.FONT_HERSHEY_SIMPLEX,
                                .7, (0,0,255), 2)

                # --- 2) Head Pose Detection ---
                pitch, yaw, roll = compute_head_pose(lm, w, h)
                if pitch is not None:
                    if self.pitch_base is None:
                        self.pitch_base = pitch
                    else:
                        self.pitch_base = EMA_ALPHA * pitch + (1 - EMA_ALPHA) * self.pitch_base

                    dev = pitch - self.pitch_base
                    now = time.time()
                    sp = self.pitch_det.update(dev, now)
                    sy = self.yaw_det.update(yaw, now)
                    sr = self.roll_det.update(roll, now)

                    F, FS, TH = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                    texts = [f"Pitch: {pitch:.2f}", f"Yaw: {yaw:.2f}", f"Roll: {roll:.2f}"]
                    ys    = [30, 60, 90]
                    for txt, y_ in zip(texts, ys):
                        tw, _ = cv2.getTextSize(txt, F, FS, TH)[0]
                        cv2.putText(frame, txt, (w - tw - 20, y_), F, FS, (0,255,0), TH)

                    alerts = [
                        (sp, "Fatigue Alert: Prolonged Head Tilt!"),
                        (sy, "Fatigue Alert: Prolonged Yaw!"),
                        (sr, "Fatigue Alert: Prolonged Roll!")
                    ]
                    for idx, (flag, msg) in enumerate(alerts):
                        if flag:
                            yy, _ = (120 + idx*30, None)
                            tw, _ = cv2.getTextSize(msg, F, FS, TH)[0]
                            cv2.putText(frame, msg, (w - tw - 20, yy), F, FS, (0,0,255), TH)

                # --- 3) Yawn Detection ---
                mar = compute_mar(lm, w, h)
                F, FS, TH = cv2.FONT_HERSHEY_SIMPLEX, .7, 2
                cv2.putText(frame, f"MAR: {mar:.2f}",
                            (30, h - 60), F, FS, TH)
                if self.yawn_det.update(mar, time.time()):
                    msg = "Fatigue Alert: Yawning Detected!"
                    cv2.putText(frame, msg, (30, h - 30),
                                F, FS, (0,0,255), TH)

                # --- Draw Face Mesh ---
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=results.multi_face_landmarks[0],
                    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=DRAWING_SPEC,
                    connection_drawing_spec=DRAWING_SPEC
                )

                # —— Immediate alerts ——
                now = time.time()

                # a) Prolonged blink > 3 s
                if self.blink_det.blink_start and (now - self.blink_det.blink_start) > 3.0:
                    self._play_sound()

                # b) Head-tilt > 5 s on any axis
                for det in (self.pitch_det, self.yaw_det, self.roll_det):
                    if det.start_time and (now - det.start_time) > 5.0:
                        self._play_sound()

                # c) Excessive blink count > 32
                if self.blink_det.blink_count > 32:
                    self._play_sound()

                # —— Accumulate score for 15 s window ——
                for m in msgs:
                    if "Fatigue Alert: Prolonged Blink!" in m:
                        self.window_score += 1.5
                if rate > 19:
                    self.window_score += 1.0
                if any((sp, sy, sr)):
                    self.window_score += 1.0
                if self.yawn_det.update(mar, now):
                    self.window_score += 2.0

                if now - self.window_start >= self.window_duration:
                    if self.window_score >= 3.5:
                        self._play_sound()
                    self.window_score = 0.0
                    self.window_start = now

            cv2.imshow("Unified Fatigue Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        self.mesh.close()
