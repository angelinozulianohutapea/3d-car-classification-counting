import cv2
from object_detector import ObjectDetector

image = cv2.imread("carla_dataset/images/image_0001.png")

if image is None:
    print("Gambar tidak ditemukan")
    exit()

detector = ObjectDetector(
    "centernet-3d-bbox.pth",
    0.2
)

boxes = detector.detect(image)

print("Jumlah deteksi:", len(boxes))

for i, box in enumerate(boxes):

    x = int(box["x"])
    y = int(box["y"])

    print(i + 1, "x =", x, "y =", y)

    cv2.circle(
        image,
        (x, y),
        8,
        (0, 0, 255),
        -1
    )

    cv2.putText(
        image,
        f"Vehicle {i + 1}",
        (x + 10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

cv2.imwrite(
    "carla_detection.jpg",
    image
)

print("Hasil disimpan: carla_detection.jpg")
