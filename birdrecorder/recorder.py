import cv2
from datetime import datetime
from pathlib import Path


class Hysteresis:
    def __init__(self, controlled_instance, start_delay=5, stop_delay=5):
        self.START_AFTER_CNT = max(0, start_delay - 1)
        self.STOP_AFTER_CNT = max(0, stop_delay - 1)
        self.start_cnt = 0
        self.stop_cnt = 0
        self.state = False
        self.controlled_instance = controlled_instance

    def step(self, condition):
        if not self.state:  # Not active -> wait for start
            if condition:
                if self.start_cnt == self.START_AFTER_CNT:
                    self.state = True
                    self.controlled_instance.start()
                self.start_cnt = self.start_cnt + 1
                self.stop_cnt = 0
            else:  # restart the start counter if we get a negative
                self.start_cnt = 0

        else:  # Active -> wait to stop
            if not condition:
                if self.stop_cnt == self.STOP_AFTER_CNT:
                    self.state = False
                    self.controlled_instance.stop()
                self.stop_cnt = self.stop_cnt + 1
                self.start_cnt = 0
            else:  # reset the stop counter
                self.stop_cnt = 0
        return self.state


class Recorder:
    def __init__(self, basepath: Path, width, height, framerate):
        self.fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.path = basepath
        self.framerate = int(framerate)
        self.width = int(width)
        self.height = int(height)
        self.writer = None

    def start(self):
        print(f"Start recording {self.width}x{self.height}@{self.framerate}")
        filename = "rec_" + datetime.now().strftime("_%y%m%d_%H:%M:%S") + ".avi"
        filename = self.path / filename
        self.writer = cv2.VideoWriter(
            filename, self.fourcc, self.framerate, (self.width, self.height)
        )

    def stop(self):
        if self.writer:
            self.writer.release()
        self.writer = None

    def append_frame(self, frame):
        if not self.writer:
            return
        date_time = datetime.now().strftime("%H:%M:%S")
        cv2.putText(
            frame,
            date_time,
            (10, 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (100, 100, 100),
            0,
        )
        self.writer.write(frame)

    def __exit__(self, exc_type, exc_value, traceback):
        self.writer.release()
