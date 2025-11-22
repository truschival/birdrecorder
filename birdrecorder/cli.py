import cv2
import argparse
from pathlib import Path

from birdrecorder.detectors import make_detector
from birdrecorder.recorder import Hysteresis, Recorder


def open_webcam(args) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        raise SystemError(f"Cannot open stream device {args.device}")

    ret, _ = cap.read()
    if not ret:
        raise SystemError(f"Reading from camera raised {ret}")

    if cap.set(cv2.CAP_PROP_AUTO_WB, int(not args.disable_auto_white_balance)):
        print(f"Auto white balance: {not args.disable_auto_white_balance}")
    if cap.set(cv2.CAP_PROP_WB_TEMPERATURE, int(args.color_temp)):
        print(f"Set color temperature to {args.color_temp}")

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
        default=500,
        help="Minimum area for motion detection (default: 500)",
    )

    parser.add_argument(
        "--device", type=int, default=0, help="Camera device for capture"
    )

    parser.add_argument(
        "--disable-auto-white-balance",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="disable auto white balance",
    )
    parser.add_argument(
        "--color-temp",
        type=int,
        default=5500,
        help="Color correction temperature (if --disable-auto-white-balance)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    detector = make_detector(args)
    cap = open_webcam(args)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera opened: {width}x{height} @ {fps}fps")
    recorder = Recorder(Path("."), width, height, fps)
    recording_hysteresis = Hysteresis(recorder, start_delay=15, stop_delay=30)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes = detector.detect(frame)
        # Did we find somthing -> add to hysteresis
        recording_hysteresis.step(len(boxes) > 0)

        # If recording, annotate the frame
        if recording_hysteresis.state:
            for box in boxes:
                box.draw_on_frame(frame)

        # The recorder will discard if inactive
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
