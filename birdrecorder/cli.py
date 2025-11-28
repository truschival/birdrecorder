import cv2
import argparse
from pathlib import Path
import numpy as np
import logging
from datetime import datetime
from birdrecorder.detectors import make_detector
from birdrecorder.recorder import Hysteresis, Recorder, CircularFrameStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("birdrecorder")


def open_stream(args) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemError(f"Cannot open stream device {args.source}")

    ret, _ = cap.read()
    if not ret:
        raise SystemError(f"Error reading from source {ret}")


    logger.info(f"Autofocus {cap.get(cv2.CAP_PROP_AUTOFOCUS)}")
    logger.info(f"Auto exposure {cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)}")
    logger.info(f"Auto white balance {cap.get(cv2.CAP_PROP_AUTO_WB)}")
    logger.info(f"White balance temperature {cap.get(cv2.CAP_PROP_WB_TEMPERATURE)}")
    logger.info(f"Brightness {cap.get(cv2.CAP_PROP_BRIGHTNESS)}")
    logger.info(f"Contrast {cap.get(cv2.CAP_PROP_CONTRAST)}")

    if cap.set(cv2.CAP_PROP_AUTOFOCUS, int(args.auto_focus)):
        logger.info(f"Setting auto focus: {args.auto_focus}")
    if cap.set(cv2.CAP_PROP_AUTO_WB, int(args.auto_white_balance)):
        logger.info(f"Setting auto white balance: {args.auto_white_balance}")
    if cap.set(cv2.CAP_PROP_WB_TEMPERATURE, int(args.color_temp)):
        logger.info(f"Setting color temperature to {args.color_temp}")

    return cap


def parse_args():
    parser = argparse.ArgumentParser(description="Motion detection and recording")

    parser.add_argument(
        "--detection",
        choices=["yolo", "opencv"],
        default="yolo",
        help="Detection method: yolo for object detection, opencv for motion detection (default: yolo)",
    )

    parser.add_argument(
        "--min-area",
        type=int,
        default=1000,
        help="Minimum area for motion detection (default: 1000)",
    )

    parser.add_argument(
        "--source", type=str, default=0, help="Camera device or video file path"
    )

    parser.add_argument(
        "--hysteresis",
        type=int,
        default=15,
        help="Number of frames with(out) motion or object until recording starts/stops (default: 15)",
    )

    parser.add_argument(
        "--auto-white-balance",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
        help="camera auto white balance",
    )

    parser.add_argument(
        "--auto-focus",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="camera auto focus",
    )

    parser.add_argument(
        "--color-temp",
        type=int,
        default=5500,
        help="Color correction temperature (if --disable-auto-white-balance)",
    )

    parser.add_argument(
        "--mark",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Mark detected objects on frame",
    )


    return parser.parse_args()


def mask_frame(frame):
    height, width = frame.shape[:2]
    mask = np.zeros((height, width), dtype="uint8")
    cv2.rectangle(
        mask,
        (int(height * 0.1), int(width * 0.1)),
        (int(height * 1.0), int(width * 0.9)),
        255,
        -1,
    )
    return cv2.bitwise_and(frame, frame, mask=mask)


def timestamp_frame(frame):
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


def main():
    args = parse_args()
    detector = make_detector(args)
    cap = open_stream(args)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    logger.info(f"Camera opened: {width}x{height} @ {fps}fps")
    recorder = Recorder(Path("."), width, height, fps, CircularFrameStore(3*args.hysteresis))
    recording_hysteresis = Hysteresis(recorder, start_delay=args.hysteresis, stop_delay=args.hysteresis)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        masked_frame = mask_frame(frame)

        boxes = detector.detect(masked_frame)
        # Did we find something -> add to hysteresis
        recording_hysteresis.step(len(boxes) > 0)

        # If recording, annotate the frame
        if recording_hysteresis.state and args.mark:
            for box in boxes:
                box.draw_on_frame(frame)

        # The recorder will discard if inactive
        timestamp_frame(frame)
        recorder.append_frame(frame)

        # Show for debugging
        cv2.imshow("Motion", frame)
        if cv2.waitKey(30) == 27:  # ESC to quit
            break

    cap.release()
    recorder.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
