import time
from config import MOUTH_AR_THRESH, SUSTAINED_TIME_THRESHOLD, ALERT_DISPLAY_DURATION

class YawnDetector:
    def __init__(self):
        self.thresh       = MOUTH_AR_THRESH
        self.sustain_sec  = SUSTAINED_TIME_THRESHOLD
        self.display_sec  = ALERT_DISPLAY_DURATION

        self.start_time   = None
        self.alerted      = False
        self.alert_time   = None

    def update(self, mar, now):
        if mar > self.thresh:
            if self.start_time is None:
                self.start_time = now
            elif (now - self.start_time >= self.sustain_sec) and not self.alerted:
                self.alerted    = True
                self.alert_time = now
        else:
            self.start_time = None
            self.alerted    = False

        if self.alerted:
            if now - self.alert_time <= self.display_sec:
                return True
            else:
                self.alerted = False
        return False
