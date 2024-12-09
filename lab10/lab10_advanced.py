from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR, draw_ocr

# Load the YOLO license plate detection model
model = YOLO("yolo11n.pt")  # Replace with a model trained for license plates

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Initialize PaddleOCR

# Open the video file
video_path = "subway.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
output_path = "output_tracking.mp4"

# Initialize VideoWriter
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Store the track history
track_history = defaultdict(lambda: {"positions": [], "class": None, "movement": None})

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO detection
        results = model.track(frame, persist=True)

        # Get the boxes
        boxes = results[0].boxes.xywh.cpu()
        confidences = results[0].boxes.conf.cpu().tolist()  # Confidence scores

        # Get the track IDs
        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
        else:
            track_ids = list(range(len(boxes)))

        # Get class labels
        if results[0].boxes.cls is not None:
            class_ids = results[0].boxes.cls.int().cpu().tolist()
        else:
            class_ids = [None] * len(track_ids)

        # Visualize results and extract license plates
        annotated_frame = results[0].plot()

        for box, track_id, class_id, confidence in zip(boxes, track_ids, class_ids, confidences):
            x, y, w, h = box
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            # Crop the detected license plate
            license_plate = frame[max(0, y-h//2):min(frame_height, y+h//2), 
                                  max(0, x-w//2):min(frame_width, x+w//2)]
            
            # Perform OCR on the cropped license plate
            if license_plate.size > 0:  # Ensure non-empty crop
                ocr_results = ocr.ocr(license_plate, cls=True)
                for line in ocr_results[0]:
                    text, confidence_score = line[1]
                    print(f"Detected Text: {text}, Confidence: {confidence_score}")
                    
                    # Annotate the text on the video frame
                    cv2.putText(
                        annotated_frame,
                        text,
                        (x - w // 2, y - h // 2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

            # Draw bounding boxes for detected objects
            cv2.rectangle(annotated_frame, 
                          (x - w // 2, y - h // 2), 
                          (x + w // 2, y + h // 2), 
                          (255, 0, 0), 2)

        # Write the frame to the output video
        out.write(annotated_frame)

        # Display the annotated frame
        cv2.namedWindow('YOLO Tracking with OCR', cv2.WINDOW_KEEPRATIO)
        cv2.imshow("YOLO Tracking with OCR", annotated_frame)
        cv2.resizeWindow('YOLO Tracking with OCR', 1240, 700)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
