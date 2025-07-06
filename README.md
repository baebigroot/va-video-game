# Virtual Assistant for Video Game Accessibility

## Overview

This script performs object detection, tracking, and OCR (Optical Character Recognition) on a video file. It is designed to detect compasses in video frames (footage from Hogwarts Legacy) using a YOLOv8 model, track their movement with the SORT algorithm, and recognize digits below the detected compasses using EasyOCR.

## Main Features

- **Object Detection:**  
  Uses a pre-trained YOLOv8 model to detect compasses in each video frame.

- **Object Tracking:**  
  Applies the SORT (Simple Online and Realtime Tracking) algorithm to assign and maintain unique IDs for each detected compass across frames.

- **OCR Recognition:**  
  Utilizes EasyOCR to read and extract digits from the region of interest (ROI) below each detected compass.

- **Result Visualization:**  
  Draws bounding boxes and IDs for each tracked compass, displays detected digits, and shows the current FPS on the video.

- **Output:**  
  Saves the processed video with all annotations to the `output/` directory.

## How It Works

1. Loads the YOLOv8 model and initializes the SORT tracker and EasyOCR reader.
2. Opens the input video file.
3. For each frame:
   - Detects compasses using YOLOv8.
   - For each detected compass, defines an ROI and applies OCR to extract digits.
   - Only compasses with detected digits are tracked.
   - Updates and maintains tracking IDs for compasses.
4. Releases resources and closes all windows after processing, or press 'q' to exit early.  
**Note:** The current model is limited in its ability to generalize to all possible compass variations in the game. Some misdetections may occur, especially when the compass moves quickly across the screen.

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- Ultralytics YOLO
- PyTorch
- EasyOCR
- NumPy
- SORT (https://github.com/abewley/sort)

## Usage
**Note:** This is a model already trained using data from Roboflow. You can access the dataset directly here: https://app.roboflow.com/hogwartslegacyva/hogwarts-legacy-va/1. Feel free to use this dataset to retrain or fine-tune the model for your own usage and project needs. Then place the trained YOLOv8 weights in `runs/detect/train/weights/best.pt`.
1. Set the path to the input video in the script (`video_path` variable).
2. Run the script:
   ```
   python detect_OCR_tracking.py
   ```
3. The output video will be saved in the `output/` directory. 
