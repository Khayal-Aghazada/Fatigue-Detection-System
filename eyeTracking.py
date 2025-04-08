import cv2
import mediapipe as mp
import numpy as np
import time
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144] 
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]

def euclidean_distance(point1, point2):
    """Calculate Euclidean distance between two 2D points."""
    return np.linalg.norm(np.array(point1) - np.array(point2))

def compute_ear(landmarks, eye_indices, image_width, image_height):
    """Compute the Eye Aspect Ratio (EAR) given facial landmarks and eye indices."""
    coords = []
    for idx in eye_indices:
        lm = landmarks[idx]
        coords.append((int(lm.x * image_width), int(lm.y * image_height)))
    
    p1, p2, p3, p4, p5, p6 = coords
    vertical1 = euclidean_distance(p2, p6)
    vertical2 = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear
ema_alpha = 0.1        
current_ema = None     
threshold_delta = 0.02 
hysteresis_frames = 3  
blink_count = 0
closed_frames = 0
blink_start_time = None
blink_durations = []
min_blink_duration = 0.2
blink_rate_window = 60  
blink_timestamps = []   
cap = cv2.VideoCapture(0)
print("Starting real-time fatigue detection based on eye tracking...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_height, image_width = frame.shape[:2]
    
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark


        left_ear = compute_ear(face_landmarks, LEFT_EYE_IDX, image_width, image_height)
        right_ear = compute_ear(face_landmarks, RIGHT_EYE_IDX, image_width, image_height)
        ear = (left_ear + right_ear) / 2.0
        

        if current_ema is None:
            current_ema = ear
        

        current_ema = ema_alpha * ear + (1 - ema_alpha) * current_ema
        

        dynamic_threshold = current_ema - threshold_delta
        

        text_ear = f"EAR: {ear:.3f}"
        text_threshold = f"Threshold: {dynamic_threshold:.3f}"
        cv2.putText(frame, text_ear, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, text_threshold, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        print(text_ear, text_threshold)
        

        if ear < dynamic_threshold:
            closed_frames += 1
            if blink_start_time is None:
                blink_start_time = time.time()
        else:
            if closed_frames >= hysteresis_frames:
                blink_duration = time.time() - blink_start_time
                blink_durations.append(blink_duration)
                blink_timestamps.append(time.time())
                blink_count += 1
                alert_message = f"Blink detected! Duration: {blink_duration:.3f} sec | Total Blinks: {blink_count}"
        
                if blink_duration >= min_blink_duration:
                    alert_message += " - Fatigue Alert: Prolonged Blink!"
                    cv2.putText(frame, "Fatigue Alert: Prolonged Blink Detected!", (30, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print(alert_message)
            closed_frames = 0
            blink_start_time = None
        

        current_time = time.time()
        blink_timestamps = [t for t in blink_timestamps if current_time - t <= blink_rate_window]
        blink_rate = len(blink_timestamps) * (60 / blink_rate_window)
        text_blink_rate = f"Blink Rate: {blink_rate:.1f} blinks/min"
        cv2.putText(frame, text_blink_rate, (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        print(text_blink_rate)
        

        if blink_rate > 19:
            fatigue_alert = "Fatigue Alert: High Blink Rate!"
            cv2.putText(frame, fatigue_alert, (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            print(fatigue_alert)
        

        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=results.multi_face_landmarks[0],
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)
    
    cv2.imshow('Real-Time Fatigue Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()   
