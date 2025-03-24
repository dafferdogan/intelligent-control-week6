from ultralytics import YOLO
import cv2

# Load model YOLOv8 Instance Segmentation
model = YOLO("best.pt")

def detect_rail_lane(image_path):
    """Mendeteksi jalur rel menggunakan YOLOv8 Instance Segmentation"""
    results = model(image_path, show=True)
    results[0].save("lane_detection_result.jpg")

# Contoh penggunaan
detect_rail_lane("dataset rel.v4i.yolov8/test/images/2025_03_21_02_37_IMG_4713_JPG.rf.3cd57315a76693cd001f62326eec3b59.jpg")