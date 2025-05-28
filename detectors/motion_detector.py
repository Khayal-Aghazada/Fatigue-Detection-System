import time
from config import SUSTAINED_TIME_THRESHOLD, ALERT_DISPLAY_DURATION

class AxisMotionDetector:
    def __init__(self, threshold, sustain=SUSTAINED_TIME_THRESHOLD, display=ALERT_DISPLAY_DURATION):
        self.threshold   = threshold
        self.sustain     = sustain
        self.display     = display
        self.start_time  = None
        self.triggered   = False
        self.trigger_time= None

    def update(self, value, now):
        if abs(value) > self.threshold:
            if self.start_time is None:
                self.start_time = now
            elif (now - self.start_time >= self.sustain) and not self.triggered:
                self.triggered   = True
                self.trigger_time= now
        else:
            self.start_time = None
            self.triggered  = False

        if self.triggered:
            if now - self.trigger_time <= self.display:
                return True
            else:
                self.triggered = False
        return False
