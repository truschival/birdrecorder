import cv2
from typing import List
from abc import ABC, abstractmethod


class ConvenienceBoundingBox:
    def __init__(self, label: str, box, color=(200, 200, 0)):
        self.label = label
        self.color = color
        # Simple coordinates tuple (x1, y1, x2, y2)
        self.x1 = int(box[0])
        self.y1 = int(box[1])
        self.x2 = int(box[2])
        self.y2 = int(box[3])

    def draw_on_frame(self, frame):
        color = self.colors[self.label] if self.label in self.color else (0, 255, 0)
        cv2.rectangle(frame, (self.x1, self.y1), (self.x2, self.y2), self.color, 2)
        cv2.putText(
            frame,
            self.label,
            (self.x1 + 12, self.y1 + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.color,
            1,
        )


class Detector(ABC):
    @abstractmethod
    def detect(self, frame) -> List[ConvenienceBoundingBox]:
        pass


class YoloDetector(Detector):
    def __init__(self, things_to_search: List[str]):
        """_summary_

        Args:
            things_to_search (dict[str,Tuple]): mapping of an object to detect and a color to
        """
        from ultralytics import YOLO
        self.model = YOLO("yolo11n.pt", task="detect")
        self.interesting_things = {
            id: name
            for id, name in self.model.names.items()
            if name in things_to_search
        }

    def _check_if_objects_can_be_detected(search: List[str]):
        pass

    def detect(self, frame) -> List[ConvenienceBoundingBox]:
        results = self.model.predict(
            frame, classes=list(self.interesting_things.keys())
        )
        boxes = []
        for result in results:
            for box in result.boxes:
                cls = box.cls.int().item()
                if cls in list(self.interesting_things):
                    label = self.interesting_things[cls]
                    box = box.xyxy[0].cpu().numpy()  # x1, y1, x2, y2
        # Handle both YOLO tensor boxes and simple coordinate tuples
        # if hasattr(box, "xyxy"):
        #     # YOLO box tensor
        #     b_int = box.xyxy[0].to(int).cpu().numpy()
        #     self.x1 = b_int[0]
        #     self.y1 = b_int[1]
        #     self.x2 = b_int[2]
        #     self.y2 = b_int[3]
                    cbb = ConvenienceBoundingBox(label, box)
                    boxes.append(cbb)
        return boxes


class CvDetector(Detector):
    """OpenCV Background removal motion detector"""

    def __init__(self, min_area, color=(200, 255, 20)):
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
        cv2.imshow("FG Mask_before", fg_mask)
        # Remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        cv2.imshow("FG Mask", fg_mask)

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

                # Pass coordinates directly as tuple (x1, y1, x2, y2)
                cbb = ConvenienceBoundingBox("motion", (x, y, x + w, y + h), self.color)
                boxes.append(cbb)

        return boxes


def make_detector(args) -> Detector:
    if args.detection == "yolo":
        print("Using YOLO object detection")
        detector = YoloDetector(["person", "bird", "cat", "book"])
    else:
        print("Using OpenCV motion detection")
        detector = CvDetector(min_area=args.min_area)

    return detector
