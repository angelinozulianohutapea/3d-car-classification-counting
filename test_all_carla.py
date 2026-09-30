import cv2
import os

from object_detector import ObjectDetector
from multi_object_detection import showbox

detector = ObjectDetector(
    "centernet-3d-bbox.pth",
    0.2
)

os.makedirs("carla_results", exist_ok=True)

for i in range(1, 4):

    filename = f"carla_dataset/images/image_{i:04d}.png"

    image = cv2.imread(filename)

    if image is None:
        print("Gambar tidak ditemukan:", filename)
        continue

    print("--------------------------------")
    print("Memproses:", filename)

    boxes = detector.detect(image)

    print("Jumlah deteksi:", len(boxes))

    result = showbox(image, boxes)

    output = f"carla_results/image_{i:04d}_3d.jpg"

    cv2.imwrite(output, result)

    print("Disimpan:", output)

print("--------------------------------")
print("SELESAI")
