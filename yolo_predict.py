from dataclasses import dataclass
from ultralytics import YOLO


@dataclass
class InputData:
	image: any
	size: int = 1024


@dataclass
class OutputData:
	classes: list
	confidence: list
	boxes: list


def predict(model=YOLO, input=InputData):
	results = model(input.image, imgsz=input.size, verbose=False)
	return OutputData(classes=[results[0].names[int(c)] for c in results[0].boxes.cls], 
			   confidence=results[0].boxes.conf,
			   boxes=results[0].boxes.xyxy)
	