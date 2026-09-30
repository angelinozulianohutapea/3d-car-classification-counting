import cv2
from object_detector import ObjectDetector

image = cv2.imread("carla_dataset/images/image_0001.png")

if image is None:
    print("Gambar tidak ditemukan")
    exit()

print("Ukuran gambar:", image.shape)

detector = ObjectDetector(
    "centernet-3d-bbox.pth",
    0.2
)

boxes = detector.detect(image)

print("==============================")
print("HASIL DETEKSI CARLA")
print("==============================")
print("Jumlah deteksi:", len(boxes))

for i, box in enumerate(boxes):
    print(i + 1, box)
