from ultralytics import YOLO

# Load the PyTorch model
model = YOLO("/home/pi/Desktop/yolo_object_detection/license_plate_detector.pt")
# Export as an NCNN format
model.export(format="ncnn")