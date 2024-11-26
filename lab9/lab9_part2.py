from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('yolov8m.pt')

# Run inference on the source
results = model.track(source='car_race.mp4',show=True, tracker="bytetrack.yaml")