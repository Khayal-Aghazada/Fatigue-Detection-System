import time
from config import (
    EMA_ALPHA,
    THRESHOLD_DELTA,
    HYSTERESIS_FRAMES,
    MIN_BLINK_DURATION,
    BLINK_RATE_WINDOW,
)

class BlinkDetector:
    def __init__(self,
                 ema_alpha=EMA_ALPHA,
                 threshold_delta=THRESHOLD_DELTA,
                 hysteresis_frames=HYSTERESIS_FRAMES,
                 min_blink_duration=MIN_BLINK_DURATION,
                 blink_rate_window=BLINK_RATE_WINDOW):
        self.alpha     = EMA_ALPHA
        self.delta     = THRESHOLD_DELTA
        self.hyst      = HYSTERESIS_FRAMES
        self.min_dur   = MIN_BLINK_DURATION
        self.rate_win  = BLINK_RATE_WINDOW

        self.current_ema     = None
        self.closed_frames   = 0
        self.blink_start     = None
        self.blink_timestamps= []
        self.blink_count     = 0

    def update(self, ear: float, now: float = None) -> tuple[float, float, list[str]]:
        """
        Process one EAR reading.
        Returns:
          - dynamic_thresh: the current threshold
          - rate: current blink-rate (blinks/min)
          - messages: list of status/alert strings
        """

        # 1) Timestamp
        if now is None:
            now = time.time()

        # 2) Initialize EMA baseline once
        if self.current_ema is None:
            self.current_ema = ear

        # 3) Provisional threshold (baseline – delta)
        provisional_thresh = self.current_ema - self.delta

        # 4) Only adapt the EMA when eye is open (ear ≥ provisional_thresh)
        if ear >= provisional_thresh:
            self.current_ema = (
                self.alpha * ear
                + (1.0 - self.alpha) * self.current_ema
            )

        # 5) Final dynamic threshold
        dynamic_thresh = self.current_ema - self.delta

        messages: list[str] = []

        # 6) Hysteresis-based blink detection
        if ear < dynamic_thresh:
            # eye is “closed” this frame
            if self.blink_start is None:
                self.blink_start = now
            self.closed_frames += 1
        else:
            # eye has reopened
            if self.closed_frames >= self.hyst:
                duration = now - self.blink_start
                self.blink_count += 1
                self.blink_timestamps.append(now)

                # build blink message
                msg = (f"Blink detected! Duration: {duration:.3f} sec"
                       f" | Total Blinks: {self.blink_count}")
                # prolonged-blink alert
                if duration >= self.min_dur:
                    msg += " - Fatigue Alert: Prolonged Blink!"
                messages.append(msg)

            # reset for next blink
            self.closed_frames = 0
            self.blink_start    = None

        # 7) Blink-rate (sliding window → blinks/min)
        cutoff = now - self.rate_win
        self.blink_timestamps = [
            t for t in self.blink_timestamps if t >= cutoff
        ]
        rate = len(self.blink_timestamps) * (60.0 / self.rate_win)
        messages.append(f"Blink Rate: {rate:.1f} blinks/min")

        # 8) High-rate fatigue alert
        if rate > 19.0:
            messages.append("Fatigue Alert: High Blink Rate!")

        # 9) Return threshold, rate, and any messages
        return dynamic_thresh, rate, messages