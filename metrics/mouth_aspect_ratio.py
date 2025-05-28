from metrics.eye_aspect_ratio import euclidean_distance
from config import MOUTH_UPPER_IDX, MOUTH_LOWER_IDX, MOUTH_LEFT_IDX, MOUTH_RIGHT_IDX


def compute_mar(landmarks, w, h):
    upper = (int(landmarks[MOUTH_UPPER_IDX].x * w), int(landmarks[MOUTH_UPPER_IDX].y * h))
    lower = (int(landmarks[MOUTH_LOWER_IDX].x * w), int(landmarks[MOUTH_LOWER_IDX].y * h))
    left  = (int(landmarks[MOUTH_LEFT_IDX].x  * w), int(landmarks[MOUTH_LEFT_IDX].y  * h))
    right = (int(landmarks[MOUTH_RIGHT_IDX].x * w), int(landmarks[MOUTH_RIGHT_IDX].y * h))
    vertical   = euclidean_distance(upper, lower)
    horizontal = euclidean_distance(left, right)
    return vertical / horizontal if horizontal != 0 else 0
