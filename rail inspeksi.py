import cv2
import numpy as np


def canny_edge_detection(image_path):
    """Mendeteksi tepi menggunakan metode Canny Edge Detection"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(img_blur, 50, 150)

    cv2.imshow("rail-segmentation", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("canny_result.jpg", edges)

    return "canny_result.jpg"

# Contoh penggunaan
canny_edge_detection("dataset rel.v4i.yolov8/test/images/2025_03_21_02_37_IMG_4713_JPG.rf.3cd57315a76693cd001f62326eec3b59.jpg")