import cv2
import time
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from typing import List
from torch import Tensor


class ConvenienceBoundingBox:
    colors = {'bird' : (200,200,0),
              'person': (200,0,200),
              'book' : (0,0,200)}

    def __init__(self,label:str, box: Tensor):
        self.label = label
        b_int = box.xyxy[0].to(int).cpu().numpy()
        self.x1 = b_int[0]
        self.y1 = b_int[1]
        self.x2 = b_int[2]
        self.y2 = b_int[3]

    def draw_on_frame(self, frame):
        color =  self.colors[self.label] if self.label in self.colors.keys() else (0,255,0)
        cv2.rectangle(frame, (self.x1, self.y1), (self.x2, self.y2), color, 2) 
        cv2.putText(frame, self.label, (self.x1+12, self.y1+12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


class Hysteresis:
    def __init__(self, controlled_instance, start_delay=10, stop_delay=20):
        self.START_AFTER_CNT =start_delay
        self.STOP_AFTER_CNT = stop_delay
        self.start_cnt = 0
        self.stop_cnt = 0
        self.state = False
        self.controlled_instance = controlled_instance

    def step(self, condition):
        if not self.state: # Not active -> wait for start
            if condition:
                self.start_cnt = min(self.START_AFTER_CNT,self.start_cnt+1)
                self.stop_cnt = 0
                if self.start_cnt == self.START_AFTER_CNT:
                    self.state = True
                    self.controlled_instance.start()
        else: # Active -> wait to stop
            if not condition:
               self.stop_cnt =  min(self.STOP_AFTER_CNT,self.stop_cnt+1) 
               self.start_cnt = 0
               self.state = not (self.stop_cnt == self.STOP_AFTER_CNT) 
               if not (self.stop_cnt == self.STOP_AFTER_CNT):
                    self.state = False
                    self.controlled_instance.stop()
        return self.state


class Recorder:
    def __init__(self,basepath: Path, width, height, framerate):
        self.fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.path = basepath
        self.framerate = int(framerate)
        self.width = int(width)
        self.height = int(height)
        self.writer = None

    def start(self):
        print(f"Start recording {self.width}x{self.height}@{self.framerate}")
        filename = "rec_" + datetime.now().strftime("_%y%m%d_%H:%M:%S")+".avi"
        filename = self.path / filename
        self.writer = cv2.VideoWriter(filename, self.fourcc,  self.framerate, (self.width, self.height))

    def stop(self):
        if self.writer:
            self.writer.release()
        self.writer = None

    def append_frame(self, frame):
        if not self.writer:
            return  
        date_time = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, date_time, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 0)
        self.writer.write(frame)

    def __exit__(self, exc_type, exc_value, traceback):
        self.writer.release()



def search_anything_interesting(model, frame,  interesting_things:dict) -> List[ConvenienceBoundingBox]:
    results = model.predict(frame, classes=list(interesting_things.keys()))
    boxes = []
    for result in results:
        for box in result.boxes:
            cls= box.cls.int().item()
            if cls in list(interesting_things):
                label = interesting_things[cls]
                cbb = ConvenienceBoundingBox(label,box)
                boxes.append(cbb)
    return boxes



def open_webcam(devid: int = 0) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemError(f"Cannot open stream device {devid}")
    
    ret, _ = cap.read()
    if not ret:
        raise SystemError(f"Reading from camera raised {ret}")

    return cap


def main():
    model = YOLO("yolo11n.pt", task='detect')
    objects_of_interest = {id : name for id,name in model.names.items() if name in ['person','bird','cat','book']}
    cap = open_webcam()
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    recorder = Recorder(Path("."), width, height, fps)
    recording_hysteresis = Hysteresis(recorder)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes = search_anything_interesting(model, frame, objects_of_interest)
        recording_hysteresis.step( len(boxes) > 0 )
        if recording_hysteresis.state :
            for box in boxes:
                box.draw_on_frame(frame)
        

        recorder.append_frame(frame)
            
        cv2.imshow('Motion', frame)
        if cv2.waitKey(30) == 27:  # ESC to quit
            break

    cap.release()
    recorder.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
