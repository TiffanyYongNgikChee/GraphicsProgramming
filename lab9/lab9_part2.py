from ultralytics import YOLO

# Detect objects from classes car only
classes = [2,3,4,5]

# Load a pretrained YOLOv8n model
model = YOLO('yolov8m.pt')

# Run inference on the source
results = model.track(source='car_races.mp4',show=True, classes=classes,tracker="bytetrack.yaml")