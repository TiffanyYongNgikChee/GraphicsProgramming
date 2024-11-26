from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Run inference on the source
results = model(source="race.mp4",show=True,conf=0.4,save=True)