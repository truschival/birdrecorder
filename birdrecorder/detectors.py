import cv2
from typing import List, Tuple
from abc import ABC, abstractmethod
from random import uniform
import logging

logger = logging.getLogger("birdrecorder.detectors")


class ConvenienceBoundingBox:
    def __init__(self, label: str, box: Tuple[int, int, int, int], color=(250, 250, 0)):
        self.label = label
        self.color = color
        # Simple coordinates tuple (x1, y1, x2, y2)
        self.x1 = box[0]
        self.y1 = box[1]
        self.x2 = box[2]
        self.y2 = box[3]

    def draw_on_frame(self, frame):
        cv2.rectangle(
            frame,
            (self.x1, self.y1),
            (self.x2, self.y2),
            color=self.color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            self.label,
            (self.x1 + 12, self.y1 + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color=self.color,
            thickness=1,
        )


class Detector(ABC):
    @abstractmethod
    def detect(self, frame) -> List[ConvenienceBoundingBox]:
        pass


class YoloDetector(Detector):
    def __init__(self, things_to_search: set[str]):
        """_summary_

        Args:
            things_to_search (dict[str,Tuple]): mapping of an object to detect and a color to
        """
        from ultralytics import YOLO

        self.model = YOLO("yolo11n.pt", task="detect")
        self._check_if_objects_can_be_detected(things_to_search)
        logger.info(f"YOLO Detector initialized for: {things_to_search}")

        self.interesting_things = {
            id: (int(uniform(0, 255)), int(uniform(0, 255)), int(uniform(0, 255)))
            for id, name in self.model.names.items()
            if name in things_to_search
        }

    def _check_if_objects_can_be_detected(self, search: set[str]):
        if not search.issubset(set(self.model.names.values())):
            missing = search.difference(set(YoloDetector().model.names.values()))
            raise ValueError(f"Cannot detect objects: {missing}")

    def detect(self, frame) -> List[ConvenienceBoundingBox]:
        results = self.model.predict(
            frame, classes=list(self.interesting_things.keys())
        )
        boxes = []
        for result in results:
            for box in result.boxes:
                cls = box.cls.int().item()
                if cls in list(self.interesting_things):
                    label = self.model.names[cls]
                    box = box.xyxy[0].int().cpu().numpy()  # x1, y1, x2, y2
                    cbb = ConvenienceBoundingBox(
                        label, box, color=self.interesting_things[cls]
                    )
                    boxes.append(cbb)
        return boxes


class CvDetector(Detector):
    """OpenCV Background removal motion detector"""

    def __init__(self, min_area, color=(0, 255, 255)):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=19
        )
        self.min_area = min_area
        self.color = color

    def detect(self, frame) -> List[ConvenienceBoundingBox]:
        """
        Detect motion areas using OpenCV background subtraction.

        Args:
            bg_subtractor: OpenCV background subtractor
            frame: Current video frame
            min_area: Minimum contour area to consider as motion (default: 500 pixels)

        Returns:
            List of ConvenienceBoundingBox objects representing motion areas
        """
        fg_mask = self.bg_subtractor.apply(frame)
        # cv2.imshow("FG Mask_before", fg_mask)
        # Remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # cv2.imshow("FG Mask", fg_mask)

        # Find contours
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                cbb = ConvenienceBoundingBox("motion", (x, y, x + w, y + h), self.color)
                boxes.append(cbb)

        return boxes


def make_detector(args) -> Detector:
    if args.detection == "yolo":
        print("Using YOLO object detection")
        detector = YoloDetector({"person", "bird", "cat", "book"})
    else:
        print("Using OpenCV motion detection")
        detector = CvDetector(min_area=args.min_area)

    return detector
