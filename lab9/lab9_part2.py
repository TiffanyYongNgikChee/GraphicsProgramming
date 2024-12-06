from ultralytics import YOLO
import cv2

# Detect objects from classes car only
classes = [2,3,4,5]

# Load a pretrained YOLOv8n model
model = YOLO('yolov8m.pt')

# Specify the window name
window_name = "YOLOv8 Detection"

# Set up OpenCV window with a custom size
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Allows resizing
cv2.resizeWindow(window_name, 1240,700)        # Set desired window size (width, height)

# Run inference on the source
results = model.track(source='car_races.mp4', show=True, classes=classes, tracker="bytetrack.yaml")

# Make sure to release OpenCV resources if needed
cv2.destroyAllWindows()