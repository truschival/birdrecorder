import cv2
import argparse
from pathlib import Path
import numpy as np
import logging
from datetime import datetime
from birdrecorder.detectors import make_detector
from birdrecorder.recorder import Hysteresis, Recorder, CircularFrameStore

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
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
    logger.info(f"Hardware device acceleration {cap.get(cv2.CAP_PROP_HW_ACCELERATION)}")
    logger.info(f"Hardware device {cap.get(cv2.CAP_PROP_HW_DEVICE)}")

    if cap.set(cv2.CAP_PROP_AUTOFOCUS, int(args.auto_focus)):
        logger.info(f"Setting auto focus: {args.auto_focus}")
    if cap.set(cv2.CAP_PROP_AUTO_WB, int(args.auto_white_balance)):
        logger.info(f"Setting auto white balance: {args.auto_white_balance}")
    if cap.set(cv2.CAP_PROP_WB_TEMPERATURE, int(args.color_temp)):
        logger.info(f"Setting color temperature to {args.color_temp}")
    if cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800):
        logger.info("Setting frame width to 800")
    if cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600):
        logger.info("Setting frame height to 600")
    return cap


def parse_args():
    parser = argparse.ArgumentParser(description="Motion detection and recording")

    parser.add_argument(
        "--detection",
        choices=["yolo", "opencv"],
        default="yolo",
        help="Detection method: 'yolo' for object detection, 'opencv' for motion detection (default: yolo)",
    )

    parser.add_argument(
        "--min-area",
        type=int,
        default=1000,
        help="Minimum area for motion detection (default: 1000)",
    )

    parser.add_argument(
        "--source",
        type=str,
        default="/dev/video0",
        help="Camera device or video file path",
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

    parser.add_argument(
        "--flip",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Flip the frame horizontally",
    )

    parser.add_argument(
        "--timing",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Time the processing steps and show timing info",
    )

    return parser.parse_args()


def mask_frame(frame):
    """Mask frame to ignore 5% borders for detection.

    Args:
        frame (Matlike): current frame
    """
    height, width = frame.shape[:2]
    mask = np.zeros((height, width), dtype="uint8")
    cv2.rectangle(
        mask,
        (int(height * 0.05), int(width * 0.05)),
        (int(height * 1.0), int(width * 0.95)),
        255,
        -1,
    )
    return cv2.bitwise_and(frame, frame, mask=mask)


def timestamp_frame(frame, counter=0):
    """Add timestamp to frame

    Args:
        frame (Matlike): current frame
    """
    date_time = datetime.now().strftime("%H:%M:%S:%f")[:-3]  # HH:MM:SS:mmm
    if counter > 0:
        date_time = f"{date_time} [{counter}]"

    cv2.putText(
        frame,
        date_time,
        (150, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (100, 255, 100),
        1,
    )


def main():
    args = parse_args()
    logger.info(f"Starting birdrecorder with {args.source=}")
    detector = make_detector(args)
    cap = open_stream(args)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    af = bool(cap.get(cv2.CAP_PROP_AUTOFOCUS))
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    mark = args.mark
    timing = args.timing

    logger.info(
        f"Camera opened: {width}x{height} @ {fps}fps total {frame_count} \n---------------"
    )
    recorder = Recorder(
        Path("."), width, height, fps, CircularFrameStore(3 * args.hysteresis)
    )
    recording_hysteresis = Hysteresis(
        recorder, start_delay=args.hysteresis, stop_delay=4 * args.hysteresis
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if args.flip:
            frame = cv2.flip(frame, 0)

        masked_frame = mask_frame(frame)

        boxes = detector.detect(masked_frame, timing=timing)
        # Did we find something -> add to hysteresis
        recording_hysteresis.step(len(boxes) > 0)

        # show detected boxes
        if mark:
            for box in boxes:
                box.draw_on_frame(frame)

        # The recorder will discard if inactive
        timestamp_frame(frame, recording_hysteresis.start_cnt)
        recorder.append_frame(frame)

        # Show for debugging
        cv2.imshow("Motion", frame)
        key_event = cv2.waitKey(2) & 0xFF  # 2ms delay

        if key_event == ord("q") or key_event == 27:  # q / ESC quit
            break
        if key_event == ord("a"):  # s settings
            af = not af
            cap.set(cv2.CAP_PROP_AUTOFOCUS, int(af))
            logger.info(f"Toggling auto focus to {af}")
        if key_event == ord("m"):  # mark bounding boxes
            mark = not mark
            logger.info(f"Toggling marking detected objects to {mark}")
        if key_event == ord("t"):  # timing information
            timing = not timing
            logger.info(f"Toggling timing detected objects to {timing}")

    cap.release()
    recorder.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
