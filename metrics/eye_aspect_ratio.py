import numpy as np

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def compute_ear(landmarks, eye_indices, w, h):
    coords = []
    for idx in eye_indices:
        lm = landmarks[idx]
        coords.append((int(lm.x * w), int(lm.y * h)))
    p1, p2, p3, p4, p5, p6 = coords
    v1 = euclidean_distance(p2, p6)
    v2 = euclidean_distance(p3, p5)
    h_ = euclidean_distance(p1, p4)
    return (v1 + v2) / (2.0 * h_)
