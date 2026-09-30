import cv2
from object_detector import ObjectDetector

image = cv2.imread("periph.jpg")

if image is None:
    print("Gambar tidak ditemukan")
    exit()

detector = ObjectDetector("centernet-3d-bbox.pth", 0.2)

boxes = detector.detect(image)

print("Jumlah deteksi:", len(boxes))
print("Hasil deteksi:")

for i, box in enumerate(boxes):
    print(i + 1, box)
