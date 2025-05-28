import cv2
import numpy as np
from config import LANDMARK_INDICES, MODEL_POINTS

def compute_head_pose(landmarks, w, h):
    pts2d = []
    for idx in LANDMARK_INDICES:
        lm = landmarks[idx]
        pts2d.append([int(lm.x * w), int(lm.y * h)])
    pts2d = np.array(pts2d, dtype=np.float32)

    focal = w
    center = (w/2, h/2)
    cam_mtx = np.array([
        [focal, 0,      center[0]],
        [0,     focal,  center[1]],
        [0,     0,           1   ]
    ], dtype="double")
    dist = np.zeros((4,1))

    ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS, pts2d, cam_mtx, dist)
    if not ok:
        return None, None, None

    R, _ = cv2.Rodrigues(rvec)
    pose = np.hstack((R, tvec))
    _, _, _, _, _, _, angles = cv2.decomposeProjectionMatrix(pose)
    return float(angles[0]), float(angles[1]), float(angles[2])
