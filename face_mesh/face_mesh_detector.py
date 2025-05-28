import mediapipe as mp
from config import MP_FACE_MESH_CONFIG

class FaceMeshDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.detector     = self.mp_face_mesh.FaceMesh(**MP_FACE_MESH_CONFIG)

    def process_frame(self, rgb_frame):
        return self.detector.process(rgb_frame)

    def close(self):
        self.detector.close()
