import cv2
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger("birdrecorder.recorder")


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


class CircularFrameStore:
    """Ring buffer for frames with overwrite on full buffer."""

    def __init__(self, size: int):
        self.size = size
        self.buffer = [None] * size
        self.write_idx = 0
        self.read_idx = 0
        self.empty = True

    def write(self, item):
        self.buffer[self.write_idx] = item
        # Did we overwrite unread data? (handle special case of empty buffer)
        if self.write_idx == self.read_idx and not self.empty:
            self.read_idx = (self.read_idx + 1) % self.size
        self.empty = False
        self.write_idx = (self.write_idx + 1) % self.size

    def read(self):
        if self.empty:
            return None

        item = self.buffer[self.read_idx]
        self.read_idx = (self.read_idx + 1) % self.size
        if self.read_idx == self.write_idx:
            self.empty = True
        return item


class Recorder:
    def __init__(
        self,
        basepath: Path,
        width,
        height,
        framerate,
        recording_buffer: CircularFrameStore = None,
    ):
        self.fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.path = basepath
        self.framerate = int(framerate)
        self.width = int(width)
        self.height = int(height)
        self.writer = None
        self.recoring_buffer = recording_buffer

    def start(self):
        print(f"Start recording {self.width}x{self.height}@{self.framerate}")
        filename = "rec_" + datetime.now().strftime("_%y%m%d_%H:%M:%S") + ".avi"
        filename = self.path / filename
        self.writer = cv2.VideoWriter(
            filename, self.fourcc, self.framerate, (self.width, self.height)
        )
        if self.recoring_buffer:
            logger.info(f"Flushing buffer of size {self.recoring_buffer.size}")
            while frame := self.recoring_buffer.read():
                self.append_frame(frame)

    def stop(self):
        if self.writer:
            self.writer.release()
        self.writer = None

    def timestamp(self, frame):
        """Add timestamp to frame

        Args:
            frame (Matlike): current frame
        """
        date_time = datetime.now().strftime("%H:%M:%S:%f")[:-3]  # HH:MM:SS:mmm
        cv2.putText(
            frame,
            date_time,
            (15, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (100, 255, 100),
            1,
        )

    def append_frame(self, frame):
        if not self.writer:
            if self.recoring_buffer:
                self.recoring_buffer.write(frame)
            return
        self.writer.write(frame)

    def __exit__(self, exc_type, exc_value, traceback):
        self.writer.release()
