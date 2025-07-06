import cv2
from ultralytics import YOLO
import torch
import easyocr  
import sys
import os
import numpy as np
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'sort'))
from sort import Sort

# Load YOLOv8 model
model = YOLO("runs/detect/train/weights/best.pt") 

# Open video
video_path = "test_videos/OCR-test2.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Can't open multiple videos")
    exit()

# Initialize SORT tracker
tracker = Sort()

# OCR reader only initialize once
reader = easyocr.Reader(['en'])


# Track IDs that have been OCR recognized with digits
confirmed_ids = set()

# Get frame size and set writer
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_video = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize output video filename (if output_1.avi already exists, then output_2.avi, ...)
i = 1
while os.path.exists(f"output/output_{i}.avi"):
    i += 1
filename = f"output/output_{i}.avi"

out = cv2.VideoWriter(filename, 
                      cv2.VideoWriter_fourcc(*'XVID'), 
                      fps_video, 
                      (frame_width, frame_height))
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # YOLO detection
    results = model(frame)
    dets = [] # To store detections with digits

    # Loop through detected bounding boxes
    for result in results:
        # Get bounding boxes and confidence
        if result.boxes is not None and result.boxes.xyxy is not None:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            for box in boxes:
                x_min, y_min, x_max, y_max = box
                # Draw bounding box for every compass (detected by YOLO) (red)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                cv2.putText(frame, "Compass", (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                # Define ROI (region of interest)
                roi_height = 3 * y_max 
                y_start_roi = y_max
                y_end_roi = min(y_max + roi_height, frame.shape[0])
                x_start_roi = x_min
                x_end_roi = x_max
                roi_image = frame[y_start_roi:y_end_roi, x_start_roi:x_end_roi]

                # OCR (Optical Character Recognition)
                texts = reader.readtext(roi_image)
                digits = [char for text in texts for char in text[1] if char.isdigit()]

                if digits:
                    # If compass has digits -> include in tracker
                    dets.append([x_min, y_min, x_max, y_max, 1.0])  # 1.0 = dummy confidence

    # Tracking only for compass with digits (if there are any)
    if dets:
        dets = np.array(dets)
        tracks = tracker.update(dets)

        for track in tracks:
            x1, y1, x2, y2, track_id = track.astype(int)

            # Confirm this ID if digits still present (for redundancy) (if there are any)
            roi = frame[y2:y2 + (y2 - y1), x1:x2]
            texts = reader.readtext(roi)
            digits = [char for text in texts for char in text[1] if char.isdigit()]
            if digits:
                confirmed_ids.add(track_id)

            # Draw bounding box + ID for confirmed compass
            if track_id in confirmed_ids:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (135, 206, 250), 2)
                cv2.putText(frame, f"Compass ID {track_id}, digits are detected", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (135, 206, 250), 2)
    else:
        tracker.update()  # maintain existing tracks

    # Draw FPS
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                

    # Show and save
    cv2.imshow("Video Detection", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
