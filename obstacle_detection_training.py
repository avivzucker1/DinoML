from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Small YOLO model
model.train(data="./datasets/data.yaml", epochs=50, imgsz=640)
