import cv2

from object_detector import ObjectDetector
from multi_object_detection import showbox


# Baca gambar
image = cv2.imread("periph.jpg")

if image is None:
    print("Gambar tidak ditemukan")
    exit()


# Load model
detector = ObjectDetector(
    "centernet-3d-bbox.pth",
    0.2
)


# Deteksi kendaraan
boxes = detector.detect(image)

print("Jumlah deteksi:", len(boxes))


# Buat 3D bounding box
result = showbox(image, boxes)


# Simpan hasil
cv2.imwrite("hasil_3d_bbox.jpg", result)

print("Hasil disimpan sebagai: hasil_3d_bbox.jpg")
