# main.py
from core.calibration       import CalibrationManager
from config                 import apply_user_profile
from core.fatigue_detection import FatigueDetection

def main():
    calib = CalibrationManager(duration_s=30)
    profile = calib.load_or_run()

    apply_user_profile(profile)

    fd = FatigueDetection()
    fd.run()

if __name__ == "__main__":
    main()
