from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11n.pt")

# Open the video file
video_path = "subway.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: {"positions": [], "class": None})

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Get the boxes
        boxes = results[0].boxes.xywh.cpu()

        # Get the track IDs, or assign default IDs if not available
        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
        else:
            track_ids = list(range(len(boxes)))  # Generate sequential IDs

        # Get class labels or set to None if not available
        if results[0].boxes.cls is not None:
            class_ids = results[0].boxes.cls.int().cpu().tolist()
        else:
            class_ids = [None] * len(track_ids)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

         # Plot the tracks //tracks_id--add on to a list. Count how many id been print out
        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track["positions"].append((float(x), float(y)))  # x, y center point
            track["class"] = class_id  # Update class for this track
            
            # Retain only 30 positions for smoother visualization
            if len(track["positions"]) > 30:
                track["positions"].pop(0)

            # Draw the tracking lines
            points = np.array(track["positions"], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

        # Display the annotated frame
        cv2.namedWindow('YOLOv11 Tracking', cv2.WINDOW_KEEPRATIO)
        cv2.imshow("YOLOv11 Tracking", annotated_frame)
        cv2.resizeWindow('YOLOv11 Tracking',1240,700)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

        # Extract unique counts per class
        class_counts = defaultdict(int)
        for track_id, data in track_history.items():
            class_counts[data["class"]] += 1

        # Print the results
        print("Detected objects per class:")
        for class_id, count in class_counts.items():
            print(f"Class {class_id}: {count} objects")

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()