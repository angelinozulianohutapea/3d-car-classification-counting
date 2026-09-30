import cv2

from object_detector import ObjectDetector


# Baca frame DLR
image = cv2.imread("dlr_frame.jpg")

if image is None:
    print("Frame tidak ditemukan")
    exit()


print("Ukuran gambar:", image.shape)


# Load model
detector = ObjectDetector(
    "centernet-3d-bbox.pth",
    0.2
)


# Deteksi
boxes = detector.detect(image)


print("==============================")
print("HASIL DETEKSI DLR")
print("==============================")
print("Jumlah deteksi:", len(boxes))

for i, box in enumerate(boxes):
    print(i + 1, box)
