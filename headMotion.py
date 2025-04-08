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

landmark_indices = [1, 152, 33, 263, 61, 291]

model_points = np.array([
    [0.0, 0.0, 0.0],         
    [0.0, -63.6, -12.5],     
    [-43.3, 32.7, -26.0],    
    [43.3, 32.7, -26.0],     
    [-28.9, -28.9, -24.1],   
    [28.9, -28.9, -24.1]     
], dtype=np.float32)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    print("Failed to open camera")
    exit()

size = frame.shape
focal_length = size[1] 
center = (size[1] / 2, size[0] / 2)
camera_matrix = np.array(
    [[focal_length, 0, center[0]],
     [0, focal_length, center[1]],
     [0, 0, 1]], dtype="double"
)

dist_coeffs = np.zeros((4, 1))

pitch_threshold = 20      
yaw_threshold   = 20      
roll_threshold  = 20      
sustained_time_threshold = 2.0 

alert_display_duration = 1.0

pitch_start_time = None
pitch_baseline = None

yaw_start_time = None
roll_start_time = None

pitch_alert_triggered = False
yaw_alert_triggered = False
roll_alert_triggered = False

pitch_alert_time = None
yaw_alert_time = None
roll_alert_time = None

print("Starting real-time head motion detection...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image_height, image_width = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        image_points = []
        for idx in landmark_indices:
            lm = landmarks[idx]
            x = int(lm.x * image_width)
            y = int(lm.y * image_height)
            image_points.append([x, y])
        image_points = np.array(image_points, dtype=np.float32)


        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs)
        if success:
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            pose_matrix = np.hstack((rotation_matrix, translation_vector))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_matrix)

            pitch = float(euler_angles[0])
            yaw   = float(euler_angles[1])
            roll  = float(euler_angles[2])
            
            cv2.putText(frame, f"Pitch: {pitch:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(frame, f"Yaw: {yaw:.2f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(frame, f"Roll: {roll:.2f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
            current_time = time.time()
            
            if pitch_baseline is None:
                pitch_baseline = pitch
            else:
                pitch_baseline = 0.1 * pitch + 0.9 * pitch_baseline

            pitch_deviation = abs(pitch - pitch_baseline)
            
            if pitch_deviation > pitch_threshold:
                if pitch_start_time is None:
                    pitch_start_time = current_time
                else:
                    elapsed = current_time - pitch_start_time
                    if elapsed >= sustained_time_threshold and not pitch_alert_triggered:
                        pitch_alert_triggered = True
                        pitch_alert_time = current_time
            else:
                pitch_start_time = None
                pitch_alert_triggered = False
                pitch_alert_time = None
                
            if pitch_alert_triggered:
                if current_time - pitch_alert_time <= alert_display_duration:
                    alert_msg = "Fatigue Alert: Prolonged Head Tilt (Pitch) Detected!"
                    cv2.putText(frame, alert_msg, (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    print(alert_msg)
                else:
                    pitch_alert_triggered = False

            if abs(yaw) > yaw_threshold:
                if yaw_start_time is None:
                    yaw_start_time = current_time
                else:
                    elapsed = current_time - yaw_start_time
                    if elapsed >= sustained_time_threshold and not yaw_alert_triggered:
                        yaw_alert_triggered = True
                        yaw_alert_time = current_time
            else:
                yaw_start_time = None
                yaw_alert_triggered = False
                yaw_alert_time = None
                
            if yaw_alert_triggered:
                if current_time - yaw_alert_time <= alert_display_duration:
                    alert_msg = "Fatigue Alert: Prolonged Yaw Detected!"
                    cv2.putText(frame, alert_msg, (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    print(alert_msg)
                else:
                    yaw_alert_triggered = False

            if abs(roll) > roll_threshold:
                if roll_start_time is None:
                    roll_start_time = current_time
                else:
                    elapsed = current_time - roll_start_time
                    if elapsed >= sustained_time_threshold and not roll_alert_triggered:
                        roll_alert_triggered = True
                        roll_alert_time = current_time
            else:
                roll_start_time = None
                roll_alert_triggered = False
                roll_alert_time = None
                
            if roll_alert_triggered:
                if current_time - roll_alert_time <= alert_display_duration:
                    alert_msg = "Fatigue Alert: Prolonged Roll Detected!"
                    cv2.putText(frame, alert_msg, (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    print(alert_msg)
                else:
                    roll_alert_triggered = False
            
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=results.multi_face_landmarks[0],
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)
    
    cv2.imshow("Real-Time Head Motion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()
