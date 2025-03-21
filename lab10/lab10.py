from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLO


# Load the YOLO11 model
model = YOLO("yolo11n.pt")

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

            # Determine movement direction
            if len(track["positions"]) >= 2:
                    # Compare the x-coordinate of the last two positions
                if track["positions"][-1][0] > track["positions"][-2][0]:
                    track["movement"] = "Left to Right"
                elif track["positions"][-1][0] < track["positions"][-2][0]:
                    track["movement"] = "Right to Left"
            
            # Retain only 30 positions for smoother visualization
            if len(track["positions"]) > 30:
                track["positions"].pop(0)

            # Draw the tracking lines
            points = np.array(track["positions"], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

            # Annotate the direction of movement
            if track["movement"] is not None:
                movement_text = track["movement"]
                cv2.putText(
                    annotated_frame,
                    movement_text,
                    (int(x - w // 2), int(y - h // 2) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
        # Write the frame to the output video
        out.write(annotated_frame)

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

    # Print movement direction of tracked objects
    print("Movement direction of tracked objects:")
    for track_id, data in track_history.items():
        print(f"Track ID {track_id}: {data['movement']}")
    

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()