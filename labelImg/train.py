from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # Load YOLOv8 pre-trained model
model.info()
