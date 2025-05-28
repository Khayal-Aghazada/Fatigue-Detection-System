import cv2
from core.fatigue_detection import FatigueDetection

IP_CAM_URL = "http://192.168.38.5:8080/video"

class IPWebcamFatigueDetection(FatigueDetection):
    def __init__(self, stream_url):
        super().__init__()

        self.cap.release()
        self.cap = cv2.VideoCapture(stream_url)

        if not self.cap.isOpened():
            raise RuntimeError("❌ Could not open IP webcam stream")

if __name__ == "__main__":
    detector = IPWebcamFatigueDetection(IP_CAM_URL)
    print("✅ IP Webcam stream connected. Starting fatigue detection...")
    detector.run()
